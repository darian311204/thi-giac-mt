import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

export interface LandmarkPoint {
  x: number
  y: number
  z: number
}

export interface FingerState {
  thumb: boolean
  index: boolean
  middle: boolean
  ring: boolean
  pinky: boolean
}

export interface DetectedHand {
  handedness: string
  score: number
  landmarks: LandmarkPoint[]
  fingers: FingerState
  sign_id: string | null
  confidence: number
}

export interface SignMetadata {
  id: string
  label_vi: string
  label_en: string
  emoji: string
  category: string
  how_to: string
  color: string
  confidence?: number
  is_custom?: boolean
}

export interface DetectionResponse {
  hands: DetectedHand[]
  connections: [number, number][]
  detected_sign: SignMetadata | null
  latency_ms: number
}

export interface HistoryRecord {
  id: number
  sign_id: string
  label_vi: string
  confidence: number
  created_at: string
}

export interface ScreenshotItem {
  filename: string
  url: string
  label_vi?: string
  created_at: string
}

const DEFAULT_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
]

function App() {
  const [activeTab, setActiveTab] = useState<'live' | 'dictionary' | 'trainer'>('live')
  const [rightSubTab, setRightSubTab] = useState<'history' | 'screenshots'>('history')
  const [backendOnline, setBackendOnline] = useState<boolean>(false)
  const [signs, setSigns] = useState<SignMetadata[]>([])
  const [selectedCategory, setSelectedCategory] = useState<string>('Tất cả')

  // Input source state: Camera or Uploaded Image/Video
  const [cameraActive, setCameraActive] = useState<boolean>(false)
  const [uploadedImageSrc, setUploadedImageSrc] = useState<string | null>(null)
  const [videoFileActive, setVideoFileActive] = useState<boolean>(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [mirrorMode, setMirrorMode] = useState<boolean>(true)
  const [showSkeleton, setShowSkeleton] = useState<boolean>(true)
  const [autoSpeak, setAutoSpeak] = useState<boolean>(true)

  // Detection telemetry state
  const [detectedHands, setDetectedHands] = useState<DetectedHand[]>([])
  const [currentSign, setCurrentSign] = useState<SignMetadata | null>(null)
  const [latencyMs, setLatencyMs] = useState<number>(0)
  const [fps, setFps] = useState<number>(0)
  const [holdProgress, setHoldProgress] = useState<number>(0)

  // Sentence Builder, History & Screenshots
  const [sentence, setSentence] = useState<SignMetadata[]>([])
  const [history, setHistory] = useState<HistoryRecord[]>([])
  const [screenshots, setScreenshots] = useState<ScreenshotItem[]>([])
  const [screenshotToast, setScreenshotToast] = useState<string | null>(null)

  // Custom Sign Trainer State
  const [capturedLandmarks, setCapturedLandmarks] = useState<LandmarkPoint[] | null>(null)
  const [customId, setCustomId] = useState<string>('')
  const [customLabelVi, setCustomLabelVi] = useState<string>('')
  const [customLabelEn, setCustomLabelEn] = useState<string>('')
  const [customEmoji, setCustomEmoji] = useState<string>('🖐️')
  const [customHowTo, setCustomHowTo] = useState<string>('')
  const [trainerMessage, setTrainerMessage] = useState<string | null>(null)

  // Refs
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const uploadedImgRef = useRef<HTMLImageElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const isWaitingResponseRef = useRef<boolean>(false)
  const frameCounterRef = useRef<{ count: number; lastTime: number }>({
    count: 0,
    lastTime: performance.now(),
  })

  // Hold-to-commit tracking refs
  const heldSignIdRef = useRef<string | null>(null)
  const heldStartTimeRef = useRef<number>(0)
  const lastCommittedSignIdRef = useRef<string | null>(null)

  const speakText = useCallback((text: string, lang: 'vi-VN' | 'en-US' = 'vi-VN') => {
    if (!('speechSynthesis' in window) || !text.trim()) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text)
    utterance.lang = lang
    utterance.rate = 1.0
    window.speechSynthesis.speak(utterance)
  }, [])

  // Fetch initial signs, history & saved screenshots from backend
  const fetchBackendData = useCallback(async () => {
    try {
      const [healthRes, signsRes, histRes, shotsRes] = await Promise.all([
        fetch('/api/health'),
        fetch('/api/signs'),
        fetch('/api/history'),
        fetch('/api/screenshots'),
      ])
      if (healthRes.ok) setBackendOnline(true)
      if (signsRes.ok) {
        const sData = await signsRes.json()
        setSigns(sData.signs || [])
      }
      if (histRes.ok) {
        const hData = await histRes.json()
        setHistory(hData.history || [])
      }
      if (shotsRes.ok) {
        const scData = await shotsRes.json()
        setScreenshots(scData.screenshots || [])
      }
    } catch {
      setBackendOnline(false)
    }
  }, [])

  useEffect(() => {
    fetchBackendData()
    const interval = setInterval(fetchBackendData, 8000)
    return () => clearInterval(interval)
  }, [fetchBackendData])

  // Record a committed sign into Sentence Builder & SQLite History
  const commitRecognizedSign = useCallback(
    async (sign: SignMetadata) => {
      setSentence((prev) => {
        if (prev.length > 0 && prev[prev.length - 1].id === sign.id) {
          return prev
        }
        return [...prev, sign]
      })

      if (autoSpeak) {
        speakText(sign.label_vi, 'vi-VN')
      }

      try {
        await fetch('/api/history', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sign_id: sign.id,
            label_vi: sign.label_vi,
            confidence: sign.confidence || 0.95,
          }),
        })
        const histRes = await fetch('/api/history')
        if (histRes.ok) {
          const hData = await histRes.json()
          setHistory(hData.history || [])
        }
      } catch {
        // Ignore transient network errors
      }
    },
    [autoSpeak, speakText]
  )

  // Handle incoming detection payload from Python Backend
  const handleDetectionResult = useCallback(
    (data: DetectionResponse) => {
      setDetectedHands(data.hands || [])
      setCurrentSign(data.detected_sign || null)
      setLatencyMs(data.latency_ms || 0)

      const now = performance.now()
      frameCounterRef.current.count += 1
      const elapsed = now - frameCounterRef.current.lastTime
      if (elapsed >= 1000) {
        setFps(Math.round((frameCounterRef.current.count * 1000) / elapsed))
        frameCounterRef.current.count = 0
        frameCounterRef.current.lastTime = now
      }

      const detected = data.detected_sign
      if (detected && (detected.confidence ?? 0.9) >= 0.78) {
        if (heldSignIdRef.current !== detected.id) {
          heldSignIdRef.current = detected.id
          heldStartTimeRef.current = now
          setHoldProgress(15)
        } else {
          const holdDuration = now - heldStartTimeRef.current
          const pct = Math.min(100, Math.round((holdDuration / 850) * 100))
          setHoldProgress(pct)
          if (holdDuration >= 850 && lastCommittedSignIdRef.current !== detected.id) {
            lastCommittedSignIdRef.current = detected.id
            commitRecognizedSign(detected)
          }
        }
      } else {
        heldSignIdRef.current = null
        lastCommittedSignIdRef.current = null
        setHoldProgress(0)
      }
    },
    [commitRecognizedSign]
  )

  // Helper to draw skeleton onto any 2D canvas context
  const renderSkeletonOnContext = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      width: number,
      height: number,
      drawGridBackground = false
    ) => {
      if (drawGridBackground) {
        ctx.fillStyle = '#070b14'
        ctx.fillRect(0, 0, width, height)
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.08)'
        ctx.lineWidth = 1
        for (let x = 0; x < width; x += 32) {
          ctx.beginPath()
          ctx.moveTo(x, 0)
          ctx.lineTo(x, height)
          ctx.stroke()
        }
        for (let y = 0; y < height; y += 32) {
          ctx.beginPath()
          ctx.moveTo(0, y)
          ctx.lineTo(width, y)
          ctx.stroke()
        }
      }

      if (!showSkeleton || detectedHands.length === 0) return

      const shouldMirror = mirrorMode && cameraActive && !videoFileActive && !uploadedImageSrc

      detectedHands.forEach((hand) => {
        const lms = hand.landmarks
        if (!lms || lms.length < 21) return

        const mapX = (x: number) => (shouldMirror ? (1 - x) * width : x * width)
        const mapY = (y: number) => y * height

        let minX = width
        let minY = height
        let maxX = 0
        let maxY = 0
        lms.forEach((pt) => {
          const px = mapX(pt.x)
          const py = mapY(pt.y)
          if (px < minX) minX = px
          if (py < minY) minY = py
          if (px > maxX) maxX = px
          if (py > maxY) maxY = py
        })

        const pad = 22
        minX = Math.max(8, minX - pad)
        minY = Math.max(8, minY - pad)
        maxX = Math.min(width - 8, maxX + pad)
        maxY = Math.min(height - 8, maxY + pad)

        ctx.save()
        ctx.strokeStyle = currentSign?.color || '#3b82f6'
        ctx.lineWidth = 2
        ctx.setLineDash([6, 4])
        ctx.strokeRect(minX, minY, maxX - minX, maxY - minY)
        ctx.restore()

        const handLabel = `${hand.handedness === 'Left' ? 'Tay Trái' : 'Tay Phải'} (${Math.round(hand.score * 100)}%)`
        ctx.save()
        ctx.font = '600 13px Inter, sans-serif'
        const textWidth = ctx.measureText(handLabel).width
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)'
        ctx.fillRect(minX, Math.max(4, minY - 28), textWidth + 16, 24)
        ctx.fillStyle = '#93c5fd'
        ctx.fillText(handLabel, minX + 8, Math.max(20, minY - 12))
        ctx.restore()

        // Skeleton connections
        ctx.save()
        ctx.strokeStyle = currentSign?.color || '#38bdf8'
        ctx.lineWidth = 3.5
        ctx.lineCap = 'round'
        DEFAULT_CONNECTIONS.forEach(([i, j]) => {
          const p1 = lms[i]
          const p2 = lms[j]
          ctx.beginPath()
          ctx.moveTo(mapX(p1.x), mapY(p1.y))
          ctx.lineTo(mapX(p2.x), mapY(p2.y))
          ctx.stroke()
        })
        ctx.restore()

        // 21 Landmark joints
        const fingertips = new Set([4, 8, 12, 16, 20])
        lms.forEach((pt, idx) => {
          const cx = mapX(pt.x)
          const cy = mapY(pt.y)
          const isTip = fingertips.has(idx)
          ctx.beginPath()
          ctx.arc(cx, cy, isTip ? 6.5 : 4.5, 0, Math.PI * 2)
          ctx.fillStyle = isTip ? '#10b981' : '#ffffff'
          ctx.fill()
          ctx.lineWidth = 2
          ctx.strokeStyle = '#0f172a'
          ctx.stroke()
        })
      })
    },
    [detectedHands, showSkeleton, mirrorMode, cameraActive, videoFileActive, uploadedImageSrc, currentSign]
  )

  // Keep overlay canvas updated
  useEffect(() => {
    const canvas = overlayCanvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width || 640
    const height = canvas.height || 400
    ctx.clearRect(0, 0, width, height)

    const isNoCamVisual = !cameraActive && !videoFileActive && !uploadedImageSrc && detectedHands.length > 0
    renderSkeletonOnContext(ctx, width, height, isNoCamVisual)
  }, [detectedHands, cameraActive, videoFileActive, uploadedImageSrc, renderSkeletonOnContext])

  // Capture Screenshot (Composite Video/Image/Grid + 21 Landmarks + Sign Info HUD)
  const takeScreenshot = async () => {
    const width = 960
    const height = 600
    const offCanvas = document.createElement('canvas')
    offCanvas.width = width
    offCanvas.height = height
    const ctx = offCanvas.getContext('2d')
    if (!ctx) return

    // 1. Draw background (Video, Uploaded Image, or Cyber Grid)
    if ((cameraActive || videoFileActive) && videoRef.current && videoRef.current.readyState >= 2) {
      ctx.save()
      if (mirrorMode && cameraActive && !videoFileActive) {
        ctx.translate(width, 0)
        ctx.scale(-1, 1)
      }
      ctx.drawImage(videoRef.current, 0, 0, width, height)
      ctx.restore()
    } else if (uploadedImageSrc && uploadedImgRef.current) {
      ctx.drawImage(uploadedImgRef.current, 0, 0, width, height)
    } else {
      renderSkeletonOnContext(ctx, width, height, true)
    }

    // 2. If video/image was drawn, overlay the skeleton on top
    if ((cameraActive || videoFileActive || uploadedImageSrc) && detectedHands.length > 0) {
      renderSkeletonOnContext(ctx, width, height, false)
    }

    // 3. Draw bottom HUD banner with sign recognition result & timestamp
    ctx.save()
    ctx.fillStyle = 'rgba(11, 15, 25, 0.88)'
    ctx.fillRect(0, height - 84, width, 84)
    ctx.strokeStyle = currentSign?.color || '#3b82f6'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(0, height - 84)
    ctx.lineTo(width, height - 84)
    ctx.stroke()

    ctx.font = '700 24px Inter, sans-serif'
    ctx.fillStyle = '#ffffff'
    const titleText = currentSign
      ? `${currentSign.emoji} ${currentSign.label_vi} (${currentSign.label_en})`
      : 'SignVision AI — Khung hình Nhận diện'
    ctx.fillText(titleText, 24, height - 44)

    ctx.font = '500 15px Inter, sans-serif'
    ctx.fillStyle = '#93c5fd'
    const subText = currentSign
      ? `Độ tin cậy: ${Math.round((currentSign.confidence || 0.96) * 100)}%  •  Nhóm: ${currentSign.category}  •  Bàn tay: ${detectedHands.length}`
      : `Bàn tay phát hiện: ${detectedHands.length}  •  MediaPipe HandLandmarker 21 3D Points`
    ctx.fillText(subText, 24, height - 18)

    const timeStr = new Date().toLocaleString('vi-VN')
    ctx.font = '600 13px monospace'
    ctx.fillStyle = '#9ca3af'
    const timeW = ctx.measureText(timeStr).width
    ctx.fillText(timeStr, width - timeW - 24, height - 20)
    ctx.restore()

    const dataUrl = offCanvas.toDataURL('image/png')

    try {
      const res = await fetch('/api/screenshots', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: dataUrl,
          sign_id: currentSign?.id || 'capture',
          label_vi: currentSign?.label_vi || 'Ảnh chụp màn hình',
        }),
      })
      if (res.ok) {
        const saved = await res.json()
        setScreenshots((prev) => [saved, ...prev])
        setRightSubTab('screenshots')
        setScreenshotToast(`📸 Đã lưu vào screenshots/${saved.filename}!`)
        setTimeout(() => setScreenshotToast(null), 3500)
      }
    } catch {
      const a = document.createElement('a')
      a.href = dataUrl
      a.download = `signvision_${Date.now()}.png`
      a.click()
    }
  }

  const handleDeleteScreenshot = async (filename: string) => {
    try {
      await fetch(`/api/screenshots/${filename}`, { method: 'DELETE' })
      setScreenshots((prev) => prev.filter((s) => s.filename !== filename))
    } catch {
      // Ignore error
    }
  }

  const stopAllStreams = useCallback(() => {
    setCameraActive(false)
    setVideoFileActive(false)
    setUploadedImageSrc(null)

    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
      videoRef.current.src = ''
    }
    setDetectedHands([])
    setCurrentSign(null)
    setHoldProgress(0)
  }, [])

  // Handle Uploaded Image or Video file
  const handleFileUpload = async (file: File) => {
    stopAllStreams()
    setCameraError(null)

    if (file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file)
      if (videoRef.current) {
        videoRef.current.srcObject = null
        videoRef.current.src = url
        videoRef.current.loop = true
        await videoRef.current.play()
        setVideoFileActive(true)
      }
      return
    }

    if (file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = async () => {
        const dataUrl = reader.result as string
        setUploadedImageSrc(dataUrl)
        try {
          const res = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl }),
          })
          if (res.ok) {
            const data: DetectionResponse = await res.json()
            handleDetectionResult(data)
            if (data.detected_sign) {
              commitRecognizedSign(data.detected_sign)
            }
          }
        } catch {
          setCameraError('Không thể gửi ảnh tới Python Backend.')
        }
      }
      reader.readAsDataURL(file)
    }
  }

  // Global Paste (Ctrl+V) handler so user can paste any image directly
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      if (!e.clipboardData) return
      for (let i = 0; i < e.clipboardData.items.length; i++) {
        const item = e.clipboardData.items[i]
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (file) {
            handleFileUpload(file)
            break
          }
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  })

  // Start Real Webcam
  const startCamera = async () => {
    stopAllStreams()
    setCameraError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 400 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.src = ''
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/detect`)
      wsRef.current = ws

      ws.onopen = () => {
        setBackendOnline(true)
        isWaitingResponseRef.current = false
      }
      ws.onmessage = (event) => {
        isWaitingResponseRef.current = false
        try {
          const data: DetectionResponse = JSON.parse(event.data)
          handleDetectionResult(data)
        } catch {
          // Ignore malformed JSON
        }
      }
      ws.onerror = () => {
        isWaitingResponseRef.current = false
      }

      setCameraActive(true)
    } catch {
      setCameraError(
        '⚠️ Không tìm thấy Webcam. Bạn có thể bấm "📁 Tải Ảnh / Video" hoặc nhấn Ctrl+V để dán ảnh bàn tay!'
      )
    }
  }

  // Frame capture loop when Real Camera OR Uploaded Video File is active
  useEffect(() => {
    if (!cameraActive && !videoFileActive) return

    const interval = setInterval(async () => {
      const video = videoRef.current
      const capCanvas = captureCanvasRef.current
      if (!video || !capCanvas || video.readyState < 2) return
      if (isWaitingResponseRef.current) return

      const ctx = capCanvas.getContext('2d')
      if (!ctx) return

      capCanvas.width = 480
      capCanvas.height = 300
      ctx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height)
      const dataUrl = capCanvas.toDataURL('image/jpeg', 0.75)

      if (overlayCanvasRef.current) {
        overlayCanvasRef.current.width = 640
        overlayCanvasRef.current.height = 400
      }

      isWaitingResponseRef.current = true

      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ image: dataUrl }))
      } else {
        try {
          const res = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl }),
          })
          if (res.ok) {
            const data: DetectionResponse = await res.json()
            handleDetectionResult(data)
          }
        } catch {
          // Ignore transient error
        } finally {
          isWaitingResponseRef.current = false
        }
      }
    }, 95)

    return () => clearInterval(interval)
  }, [cameraActive, videoFileActive, handleDetectionResult])

  useEffect(() => {
    return () => {
      stopAllStreams()
    }
  }, [stopAllStreams])

  // Save custom sign
  const handleSaveCustomSign = async (e: React.FormEvent) => {
    e.preventDefault()
    setTrainerMessage(null)

    const landmarksToSave =
      capturedLandmarks || (detectedHands.length > 0 ? detectedHands[0].landmarks : null)

    if (!landmarksToSave || landmarksToSave.length !== 21) {
      setTrainerMessage(
        '⚠️ Vui lòng bật Webcam hoặc tải ảnh có bàn tay rồi bấm "Chụp mẫu 21 điểm mốc bàn tay hiện tại" trước khi lưu!'
      )
      return
    }

    if (!customLabelVi.trim()) {
      setTrainerMessage('⚠️ Vui lòng nhập tên ký hiệu tiếng Việt.')
      return
    }

    const generatedId =
      customId.trim() ||
      customLabelVi
        .trim()
        .toLowerCase()
        .normalize('NFD')
        .replace(/[\u0300-\u036f]/g, '')
        .replace(/[^a-z0-9]+/g, '_')

    try {
      const res = await fetch('/api/custom-signs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id: generatedId,
          label_vi: customLabelVi.trim(),
          label_en: customLabelEn.trim() || customLabelVi.trim(),
          emoji: customEmoji || '🖐️',
          how_to: customHowTo.trim() || 'Cử chỉ tùy chỉnh do người dùng huấn luyện',
          landmarks: landmarksToSave,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        setSigns(data.signs || [])
        setTrainerMessage(`✅ Đã lưu ký hiệu "${customLabelVi}" vào cơ sở dữ liệu SQLite thành công!`)
        setCustomId('')
        setCustomLabelVi('')
        setCustomLabelEn('')
        setCustomHowTo('')
      } else {
        setTrainerMessage('❌ Không thể lưu ký hiệu. Vui lòng kiểm tra lại dữ liệu.')
      }
    } catch {
      setTrainerMessage('❌ Lỗi kết nối tới máy chủ Python Backend.')
    }
  }

  const handleDeleteCustomSign = async (signId: string) => {
    try {
      const res = await fetch(`/api/custom-signs/${signId}`, { method: 'DELETE' })
      if (res.ok) {
        const data = await res.json()
        setSigns(data.signs || [])
      }
    } catch {
      // Ignore error
    }
  }

  const clearHistory = async () => {
    try {
      await fetch('/api/history', { method: 'DELETE' })
      setHistory([])
    } catch {
      setHistory([])
    }
  }

  const categories = ['Tất cả', ...Array.from(new Set(signs.map((s) => s.category)))]
  const filteredSigns =
    selectedCategory === 'Tất cả'
      ? signs
      : signs.filter((s) => s.category === selectedCategory)

  const fullSentenceVi = sentence.map((s) => s.label_vi).join(' • ')
  const fullSentenceEn = sentence.map((s) => s.label_en).join(' ')
  const activeHand = detectedHands[0] || null

  return (
    <div className="app-container">
      <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*,video/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const f = e.target.files?.[0]
          if (f) handleFileUpload(f)
          e.target.value = ''
        }}
      />

      {/* Compact Top Header */}
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-icon">🤟</div>
          <div className="brand-title">
            <h1>SignVision AI</h1>
            <p>Nhận diện 36 Ngôn ngữ Ký hiệu qua Camera & MediaPipe 21 3D Landmarks</p>
          </div>
        </div>

        <nav className="nav-tabs">
          <button
            type="button"
            className={`nav-tab ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            🎥 Nhận diện Trực tiếp
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === 'dictionary' ? 'active' : ''}`}
            onClick={() => setActiveTab('dictionary')}
          >
            📖 Từ điển Ký hiệu ({signs.length})
          </button>
          <button
            type="button"
            className={`nav-tab ${activeTab === 'trainer' ? 'active' : ''}`}
            onClick={() => setActiveTab('trainer')}
          >
            ✨ Tự dạy Ký hiệu mới
          </button>
        </nav>

        <div className="telemetry-badges">
          <span className={`status-pill ${backendOnline ? 'online' : 'offline'}`}>
            <span className="status-dot" />
            {backendOnline ? 'Backend: Sẵn sàng' : 'Đang kết nối...'}
          </span>
          <span className="status-pill">⚡ {latencyMs} ms</span>
          {(cameraActive || videoFileActive) && (
            <span className="status-pill">🎞️ {fps || 10} FPS</span>
          )}
        </div>
      </header>

      {/* TAB 1: LIVE WORKSPACE (Fits 100% inside viewport height) */}
      {activeTab === 'live' && (
        <div className="workspace-grid">
          {/* Left Column: Camera Viewport + Live Sign Result Banner */}
          <div className="panel-card left-stage-card">
            <div className="panel-header">
              <div className="panel-title">
                <span>🖥️ Khung hình Camera & Khung xương 21 điểm</span>
              </div>
              <div className="btn-group">
                {!cameraActive ? (
                  <button type="button" className="btn btn-primary" onClick={startCamera}>
                    🎥 Bật Webcam
                  </button>
                ) : (
                  <button type="button" className="btn btn-danger" onClick={stopAllStreams}>
                    ⏹ Tắt Webcam
                  </button>
                )}

                <button
                  type="button"
                  className="btn"
                  onClick={() => fileInputRef.current?.click()}
                  title="Tải ảnh hoặc video bàn tay từ máy tính"
                >
                  📁 Tải Ảnh / Video
                </button>

                {(uploadedImageSrc || videoFileActive) && (
                  <button type="button" className="btn btn-danger" onClick={stopAllStreams}>
                    ✕ Đóng File
                  </button>
                )}

                <button
                  type="button"
                  className="btn btn-success"
                  onClick={takeScreenshot}
                  title="Chụp ảnh màn hình kết quả nhận diện và lưu vào thư mục screenshots/"
                >
                  📸 Chụp Screenshot
                </button>
              </div>
            </div>

            {cameraError && <div className="alert-banner">{cameraError}</div>}
            {screenshotToast && (
              <div
                className="alert-banner"
                style={{
                  borderColor: 'rgba(16, 185, 129, 0.5)',
                  background: 'rgba(16, 185, 129, 0.15)',
                  color: '#6ee7b7',
                }}
              >
                {screenshotToast}
              </div>
            )}

            {/* Viewport: Supports Webcam, Uploaded Video/Image, Drag-and-Drop, or Ctrl+V Paste */}
            <div
              className="camera-viewport"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                const file = e.dataTransfer.files?.[0]
                if (file) handleFileUpload(file)
              }}
            >
              <video
                ref={videoRef}
                className={`camera-video ${mirrorMode && cameraActive && !videoFileActive ? 'mirrored' : ''}`}
                style={{ display: cameraActive || videoFileActive ? 'block' : 'none' }}
                playsInline
                muted
              />

              {uploadedImageSrc && (
                <img
                  ref={uploadedImgRef}
                  src={uploadedImageSrc}
                  alt="Uploaded hand"
                  className="camera-video"
                  style={{ objectFit: 'contain', background: '#050811' }}
                />
              )}

              <canvas ref={overlayCanvasRef} className="camera-canvas" width={640} height={400} />

              <div className="camera-hud-top">
                <div className="hud-badge">
                  <span>
                    🖐️ Bàn tay: {detectedHands.length} •{' '}
                    {cameraActive
                      ? 'Webcam Trực tiếp'
                      : videoFileActive
                        ? 'Video Tải lên'
                        : uploadedImageSrc
                          ? 'Ảnh Tải lên / Clipboard'
                          : 'Chờ tín hiệu hình ảnh'}
                  </span>
                </div>
                <div className="hud-badge" style={{ pointerEvents: 'auto', gap: '0.5rem' }}>
                  <button
                    type="button"
                    style={{
                      background: 'none',
                      border: 'none',
                      color: showSkeleton ? '#34d399' : '#9ca3af',
                      cursor: 'pointer',
                      fontWeight: 600,
                      fontSize: '0.74rem',
                    }}
                    onClick={() => setShowSkeleton((s) => !s)}
                  >
                    {showSkeleton ? '🦴 Khung xương: BẬT' : '🦴 Khung xương: TẮT'}
                  </button>
                  <button
                    type="button"
                    style={{
                      background: 'none',
                      border: 'none',
                      color: mirrorMode ? '#93c5fd' : '#9ca3af',
                      cursor: 'pointer',
                      fontWeight: 600,
                      fontSize: '0.74rem',
                    }}
                    onClick={() => setMirrorMode((m) => !m)}
                  >
                    {mirrorMode ? '🪞 Gương: BẬT' : '🪞 Gương: TẮT'}
                  </button>
                </div>
              </div>

              {!cameraActive && !videoFileActive && !uploadedImageSrc && detectedHands.length === 0 && (
                <div className="camera-placeholder">
                  <div className="placeholder-icon">📷</div>
                  <h3>Sẵn sàng Nhận diện Ngôn ngữ Ký hiệu</h3>
                  <p style={{ color: 'var(--text-secondary)', fontSize: '0.82rem' }}>
                    Bấm <strong>"🎥 Bật Webcam"</strong> để nhận diện thời gian thực, hoặc tải ảnh/video bàn tay (hỗ trợ kéo thả & dán ảnh <code>Ctrl + V</code>).
                  </p>
                  <div className="btn-group" style={{ justifyContent: 'center', marginTop: '0.25rem' }}>
                    <button type="button" className="btn btn-primary" onClick={startCamera}>
                      🎥 Bật Webcam ngay
                    </button>
                    <button
                      type="button"
                      className="btn"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      📁 Tải Ảnh / Video (hoặc Ctrl+V)
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Live Detected Sign Banner */}
            <div className={`live-sign-banner ${currentSign ? 'detected' : ''}`}>
              <div className="sign-main-info">
                <div
                  className="sign-emoji-box"
                  style={
                    currentSign
                      ? {
                          borderColor: currentSign.color,
                          background: `${currentSign.color}22`,
                        }
                      : undefined
                  }
                >
                  {currentSign ? currentSign.emoji : '🔍'}
                </div>
                <div className="sign-text-group">
                  <h2>{currentSign ? currentSign.label_vi : 'Đang chờ nhận diện cử chỉ tay...'}</h2>
                  <div className="sub-en">
                    {currentSign
                      ? `${currentSign.label_en} • Nhóm: ${currentSign.category}`
                      : 'Hỗ trợ 36 ngôn ngữ ký hiệu thuộc 7 nhóm chủ đề'}
                  </div>
                  <div className="how-to-hint">
                    {currentSign
                      ? currentSign.how_to
                      : 'Giữ cử chỉ tay ổn định ~0.85 giây để tự động ghép từ vào câu và phát âm.'}
                  </div>
                </div>
              </div>

              <div className="confidence-meter">
                <div className="confidence-value">
                  {currentSign
                    ? `${Math.round((currentSign.confidence || 0.95) * 100)}%`
                    : '--%'}
                </div>
                <div className="progress-track" title="Độ tin cậy nhận diện từ Python Backend">
                  <div
                    className="progress-fill"
                    style={{
                      width: currentSign
                        ? `${Math.round((currentSign.confidence || 0.95) * 100)}%`
                        : '0%',
                    }}
                  />
                </div>
                <div className="hold-progress-bar" title="Thời gian giữ cử chỉ để chốt từ vào câu">
                  <div
                    className="hold-progress-fill"
                    style={{ width: `${holdProgress}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Right Column: Sentence Builder + Compact 5-Finger Telemetry + Tabbed History/Screenshots */}
          <div className="right-column-stack">
            {/* 1. Sentence Builder Card */}
            <div className="panel-card compact-card">
              <div className="panel-header">
                <div className="panel-title">
                  <span>💬 Ghép câu & Phát âm (TTS)</span>
                </div>
                <button
                  type="button"
                  className={`btn btn-sm ${autoSpeak ? 'btn-active' : ''}`}
                  onClick={() => setAutoSpeak((a) => !a)}
                >
                  {autoSpeak ? '🔊 Tự đọc: BẬT' : '🔇 Tự đọc: TẮT'}
                </button>
              </div>

              <div className="sentence-box">
                {sentence.length === 0 ? (
                  <span className="sentence-empty">
                    Các ký hiệu nhận diện được sẽ tự động ghép thành câu tại đây...
                  </span>
                ) : (
                  sentence.map((item, idx) => (
                    <span key={`${item.id}-${idx}`} className="sentence-chip">
                      <span>{item.emoji}</span>
                      <span>{item.label_vi}</span>
                    </span>
                  ))
                )}
              </div>

              <div className="controls-toolbar">
                <div className="btn-group">
                  <button
                    type="button"
                    className="btn btn-sm btn-primary"
                    disabled={sentence.length === 0}
                    onClick={() => speakText(sentence.map((s) => s.label_vi).join(', '), 'vi-VN')}
                  >
                    🔊 Đọc Tiếng Việt
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-success"
                    disabled={sentence.length === 0}
                    onClick={() => speakText(fullSentenceEn, 'en-US')}
                  >
                    🌐 Speak EN
                  </button>
                </div>
                <div className="btn-group">
                  {sentence.length > 0 && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => navigator.clipboard.writeText(fullSentenceVi)}
                      title="Sao chép câu"
                    >
                      📋 Copy
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={sentence.length === 0}
                    onClick={() => setSentence((prev) => prev.slice(0, -1))}
                  >
                    ⌫ Xóa từ
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-danger"
                    disabled={sentence.length === 0}
                    onClick={() => setSentence([])}
                  >
                    🗑️ Xóa hết
                  </button>
                </div>
              </div>
            </div>

            {/* 2. Compact Real-time Finger State Telemetry */}
            <div className="panel-card compact-card">
              <div className="panel-header">
                <div className="panel-title">
                  <span>🖐️ Trạng thái 5 Ngón tay (MediaPipe)</span>
                </div>
                <span style={{ fontSize: '0.76rem', color: '#93c5fd', fontWeight: 600 }}>
                  {activeHand
                    ? `${activeHand.handedness === 'Left' ? 'Tay Trái' : 'Tay Phải'} (${Math.round(activeHand.score * 100)}%)`
                    : 'Chưa phát hiện bàn tay'}
                </span>
              </div>

              <div className="finger-pills">
                <div className={`finger-pill ${activeHand?.fingers.thumb ? 'up' : ''}`}>
                  Cái: {activeHand ? (activeHand.fingers.thumb ? 'Duỗi ↑' : 'Gập ↓') : '--'}
                </div>
                <div className={`finger-pill ${activeHand?.fingers.index ? 'up' : ''}`}>
                  Trỏ: {activeHand ? (activeHand.fingers.index ? 'Duỗi ↑' : 'Gập ↓') : '--'}
                </div>
                <div className={`finger-pill ${activeHand?.fingers.middle ? 'up' : ''}`}>
                  Giữa: {activeHand ? (activeHand.fingers.middle ? 'Duỗi ↑' : 'Gập ↓') : '--'}
                </div>
                <div className={`finger-pill ${activeHand?.fingers.ring ? 'up' : ''}`}>
                  Áp út: {activeHand ? (activeHand.fingers.ring ? 'Duỗi ↑' : 'Gập ↓') : '--'}
                </div>
                <div className={`finger-pill ${activeHand?.fingers.pinky ? 'up' : ''}`}>
                  Út: {activeHand ? (activeHand.fingers.pinky ? 'Duỗi ↑' : 'Gập ↓') : '--'}
                </div>
              </div>
            </div>

            {/* 3. Tabbed History & Screenshot Gallery Card (Flex-1 Scrollable) */}
            <div className="panel-card flex-scroll-card">
              <div className="panel-header">
                <div className="sub-tabs-group">
                  <button
                    type="button"
                    className={`sub-tab-btn ${rightSubTab === 'history' ? 'active' : ''}`}
                    onClick={() => setRightSubTab('history')}
                  >
                    🕒 Lịch sử ({history.length})
                  </button>
                  <button
                    type="button"
                    className={`sub-tab-btn ${rightSubTab === 'screenshots' ? 'active' : ''}`}
                    onClick={() => setRightSubTab('screenshots')}
                  >
                    📸 Screenshot ({screenshots.length})
                  </button>
                </div>

                {rightSubTab === 'history' && history.length > 0 && (
                  <button type="button" className="btn btn-sm" onClick={clearHistory}>
                    Xóa lịch sử
                  </button>
                )}
                {rightSubTab === 'screenshots' && (
                  <button type="button" className="btn btn-sm btn-success" onClick={takeScreenshot}>
                    + Chụp ngay
                  </button>
                )}
              </div>

              <div className="card-scroll-body">
                {rightSubTab === 'history' ? (
                  history.length === 0 ? (
                    <p className="empty-hint-text">Chưa có lịch sử nhận diện nào được lưu.</p>
                  ) : (
                    <div className="history-list">
                      {history.map((item) => {
                        const matchedSign = signs.find((s) => s.id === item.sign_id)
                        return (
                          <div key={item.id} className="history-item">
                            <div className="history-item-left">
                              <span>{matchedSign?.emoji || '🤟'}</span>
                              <span>{item.label_vi}</span>
                            </div>
                            <div className="history-item-right">
                              <span style={{ color: '#34d399' }}>
                                {Math.round(item.confidence * 100)}%
                              </span>
                              <span>{item.created_at.slice(11, 19)}</span>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  )
                ) : screenshots.length === 0 ? (
                  <p className="empty-hint-text">
                    Nhấn <strong>"📸 Chụp Screenshot"</strong> để lưu lại khung xương 21 điểm kèm kết quả nhận diện vào thư mục <code>screenshots/</code>.
                  </p>
                ) : (
                  <div className="screenshots-grid">
                    {screenshots.map((shot) => (
                      <div key={shot.filename} className="screenshot-card">
                        <a href={shot.url} target="_blank" rel="noreferrer">
                          <img
                            src={shot.url}
                            alt={shot.filename}
                            className="screenshot-thumb"
                          />
                        </a>
                        <div className="screenshot-meta">
                          <span
                            style={{
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                            title={shot.filename}
                          >
                            {shot.created_at.slice(0, 8)}
                          </span>
                          <div className="screenshot-actions">
                            <a
                              href={shot.url}
                              download={shot.filename}
                              className="icon-btn"
                              title="Tải ảnh về máy"
                            >
                              ⬇
                            </a>
                            <button
                              type="button"
                              className="icon-btn"
                              onClick={() => handleDeleteScreenshot(shot.filename)}
                              title="Xóa ảnh"
                            >
                              ✕
                            </button>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: SIGN LANGUAGE DICTIONARY */}
      {activeTab === 'dictionary' && (
        <div className="panel-card full-tab-card">
          <div className="panel-header">
            <div className="panel-title">
              <span>📖 Từ điển 36 Cử chỉ & Ngôn ngữ Ký hiệu Thông dụng</span>
            </div>
            <div className="category-filter-bar">
              {categories.map((cat) => (
                <button
                  key={cat}
                  type="button"
                  className={`btn btn-sm ${selectedCategory === cat ? 'btn-primary' : ''}`}
                  onClick={() => setSelectedCategory(cat)}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          <div className="dictionary-grid-scroll">
            <div className="dictionary-grid">
              {filteredSigns.map((sign) => (
                <div key={sign.id} className="sign-card">
                  <div>
                    <div className="sign-card-top">
                      <div className="sign-card-title">
                        <div
                          className="sign-card-emoji"
                          style={{
                            borderColor: sign.color,
                            background: `${sign.color}20`,
                          }}
                        >
                          {sign.emoji}
                        </div>
                        <div>
                          <h3 style={{ fontSize: '0.98rem', fontWeight: 700 }}>
                            {sign.label_vi}
                          </h3>
                          <p style={{ fontSize: '0.78rem', color: '#93c5fd' }}>
                            {sign.label_en}
                          </p>
                        </div>
                      </div>
                      <span className="category-badge">{sign.category}</span>
                    </div>

                    <p className="sign-card-desc" style={{ marginTop: '0.6rem' }}>
                      <strong>Cách thực hiện:</strong> {sign.how_to}
                    </p>
                  </div>

                  <div className="sign-card-actions">
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => speakText(sign.label_vi, 'vi-VN')}
                    >
                      🔊 Nghe phát âm
                    </button>
                    {sign.is_custom && (
                      <button
                        type="button"
                        className="btn btn-sm btn-danger"
                        onClick={() => handleDeleteCustomSign(sign.id)}
                      >
                        🗑️ Xóa
                      </button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: CUSTOM SIGN TRAINER */}
      {activeTab === 'trainer' && (
        <div className="panel-card full-tab-card">
          <div className="panel-header">
            <div className="panel-title">
              <span>✨ Tự huấn luyện & Thêm Ký hiệu Mới vào Cơ sở dữ liệu (SQLite)</span>
            </div>
          </div>

          <p style={{ color: 'var(--text-secondary)', fontSize: '0.86rem' }}>
            Bật Webcam hoặc tải ảnh bàn tay ở tab <strong>Nhận diện Trực tiếp</strong>, sau đó nhấn{' '}
            <strong>"Chụp mẫu 21 điểm mốc bàn tay hiện tại"</strong> để lưu ký hiệu mới vào{' '}
            <code>gestures.db</code>.
          </p>

          <div className="controls-toolbar">
            <div className="btn-group">
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => {
                  if (detectedHands.length > 0) {
                    setCapturedLandmarks(detectedHands[0].landmarks)
                    setTrainerMessage(
                      '✅ Đã chụp thành công vector 21 điểm mốc (landmarks) từ bàn tay hiện tại!'
                    )
                  } else {
                    setTrainerMessage(
                      '⚠️ Chưa có dữ liệu bàn tay! Hãy bật Webcam hoặc tải ảnh bàn tay ở tab Nhận diện Trực tiếp trước.'
                    )
                  }
                }}
              >
                📸 Chụp mẫu 21 điểm mốc bàn tay hiện tại
              </button>
            </div>
            <span className="status-pill">
              {capturedLandmarks
                ? 'Đã có mẫu 21 Landmarks sẵn sàng lưu'
                : detectedHands.length > 0
                  ? 'Đang có dữ liệu bàn tay'
                  : 'Chưa chụp mẫu bàn tay'}
            </span>
          </div>

          {trainerMessage && <div className="alert-banner">{trainerMessage}</div>}

          <form className="trainer-form" onSubmit={handleSaveCustomSign}>
            <div className="form-field">
              <label>Tên ký hiệu (Tiếng Việt) *</label>
              <input
                type="text"
                className="form-input"
                placeholder="Ví dụ: Xin nước uống, Đói bụng, Tuyệt vời..."
                value={customLabelVi}
                onChange={(e) => setCustomLabelVi(e.target.value)}
                required
              />
            </div>

            <div className="form-field">
              <label>Tên ký hiệu (Tiếng Anh)</label>
              <input
                type="text"
                className="form-input"
                placeholder="Ví dụ: Water, Hungry, Awesome..."
                value={customLabelEn}
                onChange={(e) => setCustomLabelEn(e.target.value)}
              />
            </div>

            <div className="form-field">
              <label>Mã định danh (ID không dấu - Tùy chọn)</label>
              <input
                type="text"
                className="form-input"
                placeholder="Ví dụ: xin_nuoc"
                value={customId}
                onChange={(e) => setCustomId(e.target.value)}
              />
            </div>

            <div className="form-field">
              <label>Biểu tượng Emoji</label>
              <input
                type="text"
                className="form-input"
                placeholder="🖐️"
                value={customEmoji}
                onChange={(e) => setCustomEmoji(e.target.value)}
              />
            </div>

            <div className="form-field full-width">
              <label>Mô tả cách đặt ngón tay</label>
              <input
                type="text"
                className="form-input"
                placeholder="Mô tả ngắn gọn cách đặt bàn tay và các ngón tay..."
                value={customHowTo}
                onChange={(e) => setCustomHowTo(e.target.value)}
              />
            </div>

            <div className="form-field full-width">
              <button type="submit" className="btn btn-success" style={{ width: 'fit-content' }}>
                💾 Lưu Ký hiệu vào Cơ sở dữ liệu
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  )
}

export default App
