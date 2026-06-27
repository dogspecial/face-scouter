import { useState, useEffect, useRef, useCallback } from 'react';
import { Loader2 } from 'lucide-react';
import { FilesetResolver, FaceLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

interface FaceLandmarkerResult {
  faceBlendshapes: { categories: { categoryName: string; score: number }[] }[];
  faceLandmarks: NormalizedLandmark[][];
}

const getEmojiForScore = (score: number): string => {
  if (score === 0) return '😐';
  if (score <= 20) return '🙂';
  if (score <= 50) return '😊';
  if (score <= 80) return '😄';
  return '🤣';
};

const getMessageForScore = (score: number): string => {
  if (score === 0) return '웃어보세요!';
  if (score <= 20) return '입꼬리가 이제 살짝 올라가네요.';
  if (score <= 50) return '입꼬리가 꽤 많이 올라가 기분 좋은 미소네요.';
  if (score <= 80) return '광대까지 올라가 환한 미소네요!';
  return '정말 완벽하게 크게 웃고 계세요!';
};

export default function App() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const requestRef = useRef<number | undefined>(undefined);
  const lastVideoTimeRef = useRef<number>(-1);
  const isActiveRef = useRef<boolean>(false);

  const [isModelLoaded, setIsModelLoaded] = useState<boolean>(false);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);
  const [score, setScore] = useState<number>(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isDetectingFace, setIsDetectingFace] = useState<boolean>(false);
  const [videoSize, setVideoSize] = useState<{ width: number; height: number }>({ width: 640, height: 480 });

  useEffect(() => {
    let isMounted = true;

    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );

        const landmarker = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 1
        });

        if (isMounted) {
          faceLandmarkerRef.current = landmarker;
          setIsModelLoaded(true);
        }
      } catch (error) {
        console.error('Error loading MediaPipe:', error);
        if (isMounted) {
          setErrorMsg('MediaPipe 로딩 실패: 네트워크 연결을 확인해주세요.');
        }
      }
    };

    initMediaPipe();

    return () => {
      isMounted = false;
      stopCamera();
    };
  }, []);

  const startCamera = async () => {
    setErrorMsg(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: 640, height: 480 }
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          if (videoRef.current) {
            setVideoSize({
              width: videoRef.current.videoWidth,
              height: videoRef.current.videoHeight
            });
          }
        };
        setIsCameraActive(true);
        isActiveRef.current = true;
      }
    } catch {
      setErrorMsg('카메라 접근 권한이 없거나 카메라를 찾을 수 없습니다.');
      setIsCameraActive(false);
      isActiveRef.current = false;
    }
  };

  const stopCamera = useCallback(() => {
    isActiveRef.current = false;
    if (requestRef.current) cancelAnimationFrame(requestRef.current);

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }

    setIsCameraActive(false);
    setScore(0);
    setIsDetectingFace(false);
  }, []);

  const detectFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !faceLandmarkerRef.current || !isActiveRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;
      const results = faceLandmarkerRef.current.detectForVideo(video, performance.now()) as FaceLandmarkerResult;

      ctx?.clearRect(0, 0, canvas.width, canvas.height);

      if (results.faceBlendshapes?.length > 0) {
        setIsDetectingFace(true);
        if (ctx && results.faceLandmarks) {
          const drawingUtils = new DrawingUtils(ctx);
          results.faceLandmarks.forEach((landmarks) => {
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_LIPS, { color: "#3B82F6", lineWidth: 2 });
            drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_FACE_OVAL, { color: "#E2E8F0", lineWidth: 1 });
          });
        }

        const blendshapes = results.faceBlendshapes[0].categories;
        const getScore = (name: string) => blendshapes.find(c => c.categoryName === name)?.score || 0;

        const smileScore = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;
        const cheekScore = (getScore('cheekSquintLeft') + getScore('cheekSquintRight')) / 2;
        const combined = (smileScore * 0.7) + (cheekScore * 0.3);

        if (combined > 0.05) {
          const scoreValue = Math.min((combined - 0.05) / 0.95 * 1.6, 1.0) * 100;
          setScore(Math.round(scoreValue));
        } else {
          setScore(0);
        }
      } else {
        setIsDetectingFace(false);
        setScore(0);
      }
    }
    if (isActiveRef.current) requestRef.current = requestAnimationFrame(detectFrame);
  }, []);

  const handleVideoPlay = () => {
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
    requestRef.current = requestAnimationFrame(detectFrame);
  };

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center p-4 text-slate-100">
      <main className="w-full max-w-lg bg-slate-900 rounded-2xl shadow-2xl overflow-hidden border border-slate-800">
        <header className="p-6 text-center border-b border-slate-800 bg-slate-900/50">
          <h1 className="text-2xl font-bold">미소 점수 측정기</h1>
        </header>

        <section className="relative w-full aspect-video bg-slate-950 flex items-center justify-center overflow-hidden">
          <video ref={videoRef} onPlay={handleVideoPlay} autoPlay playsInline muted className="w-full h-full object-cover scale-x-[-1]" />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full scale-x-[-1] pointer-events-none" width={videoSize.width} height={videoSize.height} />
          {!isCameraActive && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-slate-900/80">
              {errorMsg && <p className="text-red-400 text-sm px-4 text-center">{errorMsg}</p>}
              {isModelLoaded
                ? <button onClick={startCamera} className="bg-blue-600 hover:bg-blue-500 px-6 py-3 rounded-xl font-bold transition-colors">카메라 켜기</button>
                : <div className="flex items-center gap-2 text-slate-400"><Loader2 className="animate-spin" /><span>모델 로딩 중...</span></div>
              }
            </div>
          )}
          {isCameraActive && !isDetectingFace && (
            <div className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-black/50 text-slate-300 text-xs px-3 py-1 rounded-full">
              얼굴을 카메라에 맞춰주세요
            </div>
          )}
        </section>

        <div className="p-8 text-center">
          <div className="text-6xl mb-4">{getEmojiForScore(score)}</div>
          <div className="text-4xl font-black">{score}<span className="text-xl text-slate-500">점</span></div>
          <p className="mt-2 text-slate-400">{getMessageForScore(score)}</p>
        </div>

        {isCameraActive && (
          <div className="px-8 pb-8 text-center">
            <button onClick={stopCamera} className="text-slate-500 hover:text-slate-300 text-sm transition-colors">
              카메라 끄기
            </button>
          </div>
        )}
      </main>
    </div>
  );
}
