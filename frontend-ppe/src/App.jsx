import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Camera, Shield, ShieldAlert, UploadCloud, AlertTriangle, CheckCircle, Activity, Box, Video, Radio, PlaySquare, Square, Download, Pause, Play, Bell, X, Trash2 } from 'lucide-react';

function App() {
  const [mode, setMode] = useState('IMAGE'); // IMAGE, VIDEO, STREAM
  const [selectedMedia, setSelectedMedia] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  // Video playback states
  const [videoFrames, setVideoFrames] = useState(null); // { frames: [], fps: number, report: {} }
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const frameIntervalRef = useRef(null);

  // Streaming states
  const [streamUrl, setStreamUrl] = useState('0');
  const [isStreaming, setIsStreaming] = useState(false);
  const wsRef = useRef(null);

  const fileInputRef = useRef(null);

  // Alert states
  const [alertHistory, setAlertHistory] = useState([]);
  const [toasts, setToasts] = useState([]);
  const [showAlertPanel, setShowAlertPanel] = useState(false);
  const [alertPulse, setAlertPulse] = useState(false);
  const lastAlertVerdictRef = useRef(null);
  const audioCtxRef = useRef(null);

  // Beep sound via Web Audio API
  const playAlertBeep = useCallback(() => {
    try {
      if (!audioCtxRef.current) audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = 880;
      osc.type = 'square';
      gain.gain.setValueAtTime(0.15, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.4);
    } catch (e) { /* ignore audio errors */ }
  }, []);

  // Fire alert when UNSAFE verdict detected
  const triggerAlert = useCallback((reportData) => {
    if (!reportData || reportData.scene_verdict !== 'UNSAFE') {
      lastAlertVerdictRef.current = reportData?.scene_verdict || null;
      return;
    }
    // Only fire once per UNSAFE transition (not on every frame)
    if (lastAlertVerdictRef.current === 'UNSAFE') return;
    lastAlertVerdictRef.current = 'UNSAFE';

    // Collect violations
    const violations = reportData.workers
      .filter(w => !w.compliant)
      .map(w => `Worker ${w.worker_id}: Missing ${w.violations.join(', ')}`);

    // Add to history
    const alert = {
      id: Date.now(),
      timestamp: new Date().toLocaleTimeString(),
      verdict: 'UNSAFE',
      violations,
      mode,
    };
    setAlertHistory(prev => [alert, ...prev].slice(0, 50));

    // Show toast
    const toastId = Date.now();
    setToasts(prev => [...prev, { id: toastId, violations }]);
    setTimeout(() => setToasts(prev => prev.filter(t => t.id !== toastId)), 5000);

    // Red pulse
    setAlertPulse(true);
    setTimeout(() => setAlertPulse(false), 3000);

    // Beep
    playAlertBeep();
  }, [mode, playAlertBeep]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    };
  }, []);

  // Watch results for alert triggering
  useEffect(() => {
    if (results) triggerAlert(results);
  }, [results]);

  // Video frame playback engine
  useEffect(() => {
    if (!videoFrames || !isPlaying) {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
        frameIntervalRef.current = null;
      }
      return;
    }

    const interval = 1000 / (videoFrames.fps || 8);
    let idx = currentFrameIdx;

    frameIntervalRef.current = setInterval(() => {
      idx = (idx + 1) % videoFrames.frames.length;
      setCurrentFrameIdx(idx);
      setSelectedMedia(videoFrames.frames[idx]);
      // Update the report panel to match the current frame
      if (videoFrames.reports && videoFrames.reports[idx]) {
        setResults(processReport(videoFrames.reports[idx]));
      }
    }, interval);

    return () => {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    };
  }, [videoFrames, isPlaying]);

  const handleModeChange = (newMode) => {
    if (isStreaming) {
      wsRef.current?.close();
      setIsStreaming(false);
    }
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    setMode(newMode);
    setSelectedMedia(null);
    setResults(null);
    setVideoFrames(null);
    setCurrentFrameIdx(0);
    setIsPlaying(false);
  };

  const handleUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    // Reset the input so the same file can be re-uploaded
    event.target.value = '';

    if (mode === 'IMAGE') {
      const imageUrl = URL.createObjectURL(file);
      setSelectedMedia(imageUrl);
      setResults(null);
      analyzeImage(file);
    } else if (mode === 'VIDEO') {
      setSelectedMedia(null);
      setResults(null);
      setVideoFrames(null);
      setIsPlaying(false);
      analyzeVideo(file);
    }
  };

  const triggerUpload = () => {
    fileInputRef.current?.click();
  };

  const processReport = (report) => {
    if (!report) return null;
    const missingSet = new Set();
    let unsafeCount = 0;
    report.workers.forEach(w => {
      if (!w.compliant) unsafeCount++;
      w.violations.forEach(v => missingSet.add(v));
    });

    return {
      scene_verdict: report.scene_verdict,
      total_workers: report.total_workers,
      safe_workers: report.compliant_workers,
      unsafe_workers: unsafeCount,
      missing_equipment: Array.from(missingSet),
      workers: report.workers
    };
  };

  const analyzeImage = async (file) => {
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        alert(errData.detail || "Server error — could not analyze the image.");
        return;
      }

      const data = await response.json();
      setResults(processReport(data.report));
      setSelectedMedia(data.annotated_image);
    } catch (error) {
      console.error("Error analyzing image:", error);
      alert("Could not connect to the backend. Is the server running?");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const analyzeVideo = async (file) => {
    setIsAnalyzing(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/analyze_video", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        alert(errData.detail || "Server error — could not analyze the video.");
        return;
      }

      const data = await response.json();
      // data = { frames: [base64...], reports: [{...}, ...], fps: number }
      if (data.frames && data.frames.length > 0) {
        setVideoFrames(data);
        setSelectedMedia(data.frames[0]);
        setCurrentFrameIdx(0);
        // Show the first frame's report
        if (data.reports && data.reports[0]) {
          setResults(processReport(data.reports[0]));
        }
        setIsPlaying(true);
      } else {
        alert("No frames were processed from the video.");
      }
    } catch (error) {
      console.error("Error analyzing video:", error);
      alert("Could not connect to the backend.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleVideoPlayback = () => {
    setIsPlaying(prev => !prev);
  };

  const downloadCurrentFrame = () => {
    if (!selectedMedia) return;
    const link = document.createElement('a');
    link.href = selectedMedia;
    link.download = `ppe_frame_${currentFrameIdx}.jpg`;
    link.click();
  };

  const toggleStream = () => {
    if (isStreaming) {
      wsRef.current?.close();
      setIsStreaming(false);
    } else {
      setIsStreaming(true);
      setSelectedMedia(null);
      setResults(null);

      const ws = new WebSocket("ws://localhost:8000/ws/stream");
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(JSON.stringify({ source: streamUrl }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.error) {
          alert(data.error);
          setIsStreaming(false);
          ws.close();
          return;
        }
        setSelectedMedia(data.annotated_image);
        setResults(processReport(data.report));
      };

      ws.onerror = (err) => {
        console.error("WebSocket error", err);
        setIsStreaming(false);
      };

      ws.onclose = () => {
        setIsStreaming(false);
      };
    }
  };

  return (
    <>
    <div className="min-h-screen p-6 flex flex-col">
      {/* Header */}
      <header className="glass-panel rounded-2xl p-4 mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-neon-blue/10 p-2 rounded-lg border border-neon-blue/30">
            <Shield className="w-6 h-6 text-neon-blue" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-wide text-white m-0 leading-tight">Safety Monitoring</h1>
            <p className="text-xs text-slate-400 font-mono tracking-widest uppercase m-0">AI based PPE Detection System</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-safe-green opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-safe-green"></span>
            </span>
            <span className="text-sm text-slate-300 font-mono">SYSTEM ONLINE</span>
          </div>
          {/* Alert Bell */}
          <button
            onClick={() => setShowAlertPanel(!showAlertPanel)}
            className="relative p-2 rounded-lg border border-white/10 hover:border-alert-red/40 hover:bg-alert-red/10 transition-all"
          >
            <Bell className="w-5 h-5 text-slate-300" />
            {alertHistory.length > 0 && (
              <span className="absolute -top-1 -right-1 bg-alert-red text-white text-[10px] font-bold w-5 h-5 rounded-full flex items-center justify-center">
                {alertHistory.length > 9 ? '9+' : alertHistory.length}
              </span>
            )}
          </button>
        </div>
      </header>

      {/* Mode Selector */}
      <div className="flex justify-center mb-6">
        <div className="glass-panel rounded-full p-1 flex">
          <button onClick={() => handleModeChange('IMAGE')} className={`px-6 py-2 rounded-full text-sm font-bold tracking-wider transition-all flex items-center gap-2 ${mode === 'IMAGE' ? 'bg-neon-blue text-white shadow-[0_0_15px_rgba(0,243,255,0.4)]' : 'text-slate-400 hover:text-white'}`}>
            <Camera className="w-4 h-4" /> IMAGE
          </button>
          <button onClick={() => handleModeChange('VIDEO')} className={`px-6 py-2 rounded-full text-sm font-bold tracking-wider transition-all flex items-center gap-2 ${mode === 'VIDEO' ? 'bg-neon-blue text-white shadow-[0_0_15px_rgba(0,243,255,0.4)]' : 'text-slate-400 hover:text-white'}`}>
            <Video className="w-4 h-4" /> VIDEO
          </button>
          <button onClick={() => handleModeChange('STREAM')} className={`px-6 py-2 rounded-full text-sm font-bold tracking-wider transition-all flex items-center gap-2 ${mode === 'STREAM' ? 'bg-neon-blue text-white shadow-[0_0_15px_rgba(0,243,255,0.4)]' : 'text-slate-400 hover:text-white'}`}>
            <Radio className="w-4 h-4" /> STREAM
          </button>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main View Area */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className={`glass-panel rounded-2xl p-1 relative flex-1 flex flex-col overflow-hidden min-h-[400px] ${alertPulse ? 'alert-pulse-border' : ''}`}>
            {/* Viewport Header */}
            <div className="absolute top-4 left-4 right-4 flex justify-between items-center z-10">
              <div className="bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-md border border-white/10 flex items-center gap-2">
                <Camera className="w-4 h-4 text-slate-300" />
                <span className="text-xs font-mono text-slate-300">
                  {mode === 'STREAM' ? 'LIVE_FEED' : mode === 'VIDEO' && videoFrames ? `FRAME ${currentFrameIdx + 1}/${videoFrames.frames.length}` : 'CAM_01_INPUT'}
                </span>
              </div>
              {/* Video playback indicator */}
              {mode === 'VIDEO' && videoFrames && (
                <div className="bg-black/50 backdrop-blur-md px-3 py-1.5 rounded-md border border-white/10 flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-safe-green animate-pulse' : 'bg-orange-500'}`}></span>
                  <span className="text-xs font-mono text-slate-300">{isPlaying ? 'PLAYING' : 'PAUSED'}</span>
                </div>
              )}
            </div>

            {/* Content Area */}
            <div className="flex-1 rounded-xl bg-black/40 border border-white/5 flex items-center justify-center relative overflow-hidden group">
              {selectedMedia && (
                <img src={selectedMedia} alt="Analysis" className="w-full h-full object-contain" />
              )}

              {isAnalyzing && (
                <div className="absolute inset-0 bg-black/60 flex flex-col items-center justify-center backdrop-blur-sm">
                  <Activity className="w-12 h-12 text-neon-blue animate-pulse mb-4" />
                  <div className="font-mono text-neon-blue tracking-widest text-sm">
                    {mode === 'VIDEO' ? 'PROCESSING VIDEO FRAMES...' : 'ANALYZING SCENE...'}
                  </div>
                  <div className="text-xs text-slate-500 font-mono mt-2">This may take a moment for longer videos</div>
                </div>
              )}

              {!selectedMedia && !isAnalyzing && mode !== 'STREAM' && (
                <div onClick={triggerUpload} className="flex flex-col items-center justify-center cursor-pointer opacity-60 group-hover:opacity-100 transition-opacity">
                  <div className="bg-white/5 p-4 rounded-full border border-white/10 mb-4 group-hover:border-neon-blue/50 group-hover:text-neon-blue transition-colors">
                    {mode === 'IMAGE' ? <UploadCloud className="w-8 h-8" /> : <Video className="w-8 h-8" />}
                  </div>
                  <span className="text-sm font-medium tracking-wide">Upload {mode === 'IMAGE' ? 'Image' : 'Video'} for Analysis</span>
                  <span className="text-xs font-mono text-slate-500 mt-2">{mode === 'IMAGE' ? 'JPG, PNG supported' : 'MP4, AVI supported'}</span>
                </div>
              )}

              {!selectedMedia && !isAnalyzing && mode === 'STREAM' && !isStreaming && (
                <div className="flex flex-col items-center justify-center opacity-60">
                  <div className="bg-white/5 p-4 rounded-full border border-white/10 mb-4">
                    <Radio className="w-8 h-8" />
                  </div>
                  <span className="text-sm font-medium tracking-wide mb-2">Connect to Live Stream</span>
                  <span className="text-xs font-mono text-slate-500 max-w-xs text-center">Enter an RTSP URL below or use '0' for local webcam, then click Connect.</span>
                </div>
              )}

              {/* Hidden file input */}
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept={mode === 'IMAGE' ? "image/*" : "video/*"}
                onChange={handleUpload}
              />
            </div>

            {/* Action Bar */}
            <div className="p-4 flex justify-center gap-3">
              {mode === 'IMAGE' && (
                <button
                  onClick={triggerUpload}
                  disabled={isAnalyzing}
                  className="bg-white/10 hover:bg-white/20 border border-white/20 px-6 py-2 rounded-lg font-mono text-sm transition-all hover:border-neon-blue/50 hover:text-neon-blue disabled:opacity-50"
                >
                  {selectedMedia ? 'UPLOAD NEW IMAGE' : 'SELECT IMAGE'}
                </button>
              )}

              {mode === 'VIDEO' && (
                <>
                  <button
                    onClick={triggerUpload}
                    disabled={isAnalyzing}
                    className="bg-white/10 hover:bg-white/20 border border-white/20 px-6 py-2 rounded-lg font-mono text-sm transition-all hover:border-neon-blue/50 hover:text-neon-blue disabled:opacity-50"
                  >
                    {videoFrames ? 'UPLOAD NEW VIDEO' : 'SELECT VIDEO'}
                  </button>
                  {videoFrames && (
                    <>
                      <button
                        onClick={toggleVideoPlayback}
                        className="bg-neon-blue/20 text-neon-blue border border-neon-blue/50 px-4 py-2 rounded-lg font-mono text-sm transition-all hover:bg-neon-blue/30 flex items-center gap-2"
                      >
                        {isPlaying ? <><Pause className="w-4 h-4" /> PAUSE</> : <><Play className="w-4 h-4" /> PLAY</>}
                      </button>
                      <button
                        onClick={downloadCurrentFrame}
                        className="bg-white/10 hover:bg-white/20 border border-white/20 px-4 py-2 rounded-lg font-mono text-sm transition-all hover:border-neon-blue/50 hover:text-neon-blue flex items-center gap-2"
                      >
                        <Download className="w-4 h-4" /> SAVE FRAME
                      </button>
                    </>
                  )}
                </>
              )}

              {mode === 'STREAM' && (
                <div className="flex w-full max-w-lg gap-2">
                  <input
                    type="text"
                    value={streamUrl}
                    onChange={(e) => setStreamUrl(e.target.value)}
                    disabled={isStreaming}
                    placeholder="rtsp://... or 0 for webcam"
                    className="flex-1 bg-black/50 border border-white/10 rounded-lg px-4 py-2 font-mono text-sm text-white focus:outline-none focus:border-neon-blue/50"
                  />
                  <button
                    onClick={toggleStream}
                    className={`px-6 py-2 rounded-lg font-mono text-sm transition-all flex items-center gap-2 ${isStreaming ? 'bg-alert-red/20 text-alert-red border border-alert-red/50 hover:bg-alert-red/30' : 'bg-neon-blue/20 text-neon-blue border border-neon-blue/50 hover:bg-neon-blue/30'}`}
                  >
                    {isStreaming ? <><Square className="w-4 h-4"/> DISCONNECT</> : <><PlaySquare className="w-4 h-4"/> CONNECT</>}
                  </button>
                </div>
              )}
            </div>

            {/* Video progress bar */}
            {mode === 'VIDEO' && videoFrames && (
              <div className="px-4 pb-3">
                <input
                  type="range"
                  min={0}
                  max={videoFrames.frames.length - 1}
                  value={currentFrameIdx}
                  onChange={(e) => {
                    const idx = parseInt(e.target.value);
                    setCurrentFrameIdx(idx);
                    setSelectedMedia(videoFrames.frames[idx]);
                    // Update report for the scrubbed-to frame
                    if (videoFrames.reports && videoFrames.reports[idx]) {
                      setResults(processReport(videoFrames.reports[idx]));
                    }
                    setIsPlaying(false);
                  }}
                  className="w-full h-1 bg-white/10 rounded-full appearance-none cursor-pointer accent-neon-blue [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-neon-blue"
                />
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Analytics */}
        <div className="flex flex-col gap-6">
          {/* Status Card */}
          <div className="glass-panel rounded-2xl p-6 relative overflow-hidden">
            <div className="absolute -right-6 -top-6 w-32 h-32 bg-neon-blue/10 rounded-full blur-2xl"></div>
            <h2 className="text-sm font-mono text-slate-400 mb-6 flex items-center gap-2">
              <Activity className="w-4 h-4" />
              COMPLIANCE STATUS
            </h2>

            {results ? (
              <div className="flex items-start gap-4">
                {results.scene_verdict === "SAFE" ? (
                  <div className="bg-safe-green/20 p-3 rounded-xl border border-safe-green/30">
                    <CheckCircle className="w-8 h-8 text-safe-green" />
                  </div>
                ) : (
                  <div className="bg-alert-red/20 p-3 rounded-xl border border-alert-red/30">
                    <ShieldAlert className="w-8 h-8 text-alert-red" />
                  </div>
                )}
                <div>
                  <div className={`text-2xl font-bold tracking-wider ${results.scene_verdict === 'SAFE' ? 'text-safe-green' : 'text-alert-red'}`}>
                    {results.scene_verdict}
                  </div>
                  <p className="text-sm text-slate-400 mt-1">
                    {results.scene_verdict === 'SAFE'
                      ? 'All personnel are fully equipped.'
                      : 'Safety violations detected in area.'}
                  </p>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 font-mono text-sm">Awaiting input...</div>
            )}
          </div>

          {/* Worker Details Card */}
          <div className="glass-panel rounded-2xl p-6 flex-1 flex flex-col min-h-[400px]">
            <h2 className="text-sm font-mono text-slate-400 mb-6 flex items-center gap-2">
              <Box className="w-4 h-4" />
              WORKER DETAILS
            </h2>

            {results ? (
              <div className="flex-1 overflow-y-auto pr-2 space-y-4">
                {results.workers.map((worker) => {
                  const status = worker.compliant && worker.alerts.length === 0 ? "SAFE" :
                    worker.compliant ? "ALERT" : "UNSAFE";
                  const borderColor = status === "SAFE" ? "border-safe-green/30" :
                    status === "ALERT" ? "border-orange-500/30" : "border-alert-red/30";
                  const bgColor = status === "SAFE" ? "bg-safe-green/5" :
                    status === "ALERT" ? "bg-orange-500/5" : "bg-alert-red/5";

                  return (
                    <div key={worker.worker_id} className={`border ${borderColor} ${bgColor} rounded-xl p-4`}>
                      <div className="flex items-center justify-between mb-3">
                        <div className="font-mono font-bold text-white flex items-center gap-2">
                          <span className={`w-2 h-2 rounded-full ${status === 'SAFE' ? 'bg-safe-green' : status === 'ALERT' ? 'bg-orange-500' : 'bg-alert-red'}`}></span>
                          WORKER {worker.worker_id}
                        </div>
                        <div className={`text-xs font-bold px-2 py-0.5 rounded ${status === 'SAFE' ? 'bg-safe-green/20 text-safe-green' : status === 'ALERT' ? 'bg-orange-500/20 text-orange-500' : 'bg-alert-red/20 text-alert-red'}`}>
                          {status}
                        </div>
                      </div>

                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-slate-400">Detected:</span>
                          <span className="text-slate-200">{worker.ppe_detected.length > 0 ? worker.ppe_detected.join(', ') : 'None'}</span>
                        </div>
                        {worker.violations.length > 0 && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Missing:</span>
                            <span className="text-alert-red font-medium">{worker.violations.join(', ')}</span>
                          </div>
                        )}
                        {worker.alerts.length > 0 && (
                          <div className="flex justify-between">
                            <span className="text-slate-400">Advisory:</span>
                            <span className="text-orange-500 font-medium">{worker.alerts.join(', ')}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm border border-dashed border-white/10 rounded-xl">
                No data available
              </div>
            )}
          </div>
        </div>
      </div>
    </div>

      {/* Toast Notifications */}
      <div className="fixed top-6 right-6 z-50 flex flex-col gap-3 max-w-sm">
        {toasts.map((toast) => (
          <div key={toast.id} className="toast-enter glass-panel rounded-xl p-4 border border-alert-red/40 bg-alert-red/10">
            <div className="flex items-center gap-2 mb-2">
              <AlertTriangle className="w-4 h-4 text-alert-red" />
              <span className="text-sm font-bold text-alert-red font-mono">SAFETY VIOLATION</span>
              <button onClick={() => setToasts(prev => prev.filter(t => t.id !== toast.id))} className="ml-auto">
                <X className="w-3 h-3 text-slate-400 hover:text-white" />
              </button>
            </div>
            {toast.violations.map((v, i) => (
              <div key={i} className="text-xs text-slate-300 font-mono ml-6">{v}</div>
            ))}
          </div>
        ))}
      </div>

      {/* Alert History Panel */}
      {showAlertPanel && (
        <div className="fixed inset-0 z-40" onClick={() => setShowAlertPanel(false)}>
          <div className="absolute top-20 right-6 w-96 max-h-[70vh] alert-panel-enter" onClick={(e) => e.stopPropagation()}>
            <div className="glass-panel rounded-2xl border border-white/10 flex flex-col overflow-hidden">
              <div className="p-4 flex items-center justify-between border-b border-white/10">
                <div className="flex items-center gap-2">
                  <Bell className="w-4 h-4 text-alert-red" />
                  <span className="font-mono text-sm font-bold text-white">ALERT HISTORY</span>
                  <span className="text-xs text-slate-400">({alertHistory.length})</span>
                </div>
                <div className="flex items-center gap-2">
                  {alertHistory.length > 0 && (
                    <button onClick={() => setAlertHistory([])} className="text-xs text-slate-400 hover:text-alert-red transition-colors flex items-center gap-1">
                      <Trash2 className="w-3 h-3" /> Clear
                    </button>
                  )}
                  <button onClick={() => setShowAlertPanel(false)}>
                    <X className="w-4 h-4 text-slate-400 hover:text-white" />
                  </button>
                </div>
              </div>
              <div className="overflow-y-auto max-h-[60vh] p-3 space-y-2">
                {alertHistory.length === 0 ? (
                  <div className="text-center text-slate-500 font-mono text-sm py-8">No alerts yet</div>
                ) : (
                  alertHistory.map((alert) => (
                    <div key={alert.id} className="border border-alert-red/20 bg-alert-red/5 rounded-lg p-3">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-xs font-bold text-alert-red font-mono">UNSAFE</span>
                        <span className="text-xs text-slate-500 font-mono">{alert.timestamp} • {alert.mode}</span>
                      </div>
                      {alert.violations.map((v, i) => (
                        <div key={i} className="text-xs text-slate-300 font-mono">{v}</div>
                      ))}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
