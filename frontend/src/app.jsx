import React, {
  useEffect,
  useRef,
  useState,
} from "react";

import {
  loginUser,
  registerUser,
  logoutUser,
} from "./api/authApi";

import {
  triggerSOS,
  getEmergencyContacts,
  addEmergencyContact,
  deleteEmergencyContact,
} from "./api/emergencyApi";

import {
  getMonitoringStatus,
  startMonitoring,
  stopMonitoring,
  detectCamera,
  analyzeSafety,
} from "./api/monitoringApi";

import { getHistory } from "./api/historyApi";

import "./dashboard.css";

function App() {
  // ==========================================
  // AUTH
  // ==========================================

  const [mode, setMode] = useState("login");

  const [loggedIn, setLoggedIn] = useState(
    !!localStorage.getItem("access_token")
  );

  const [form, setForm] = useState({
    name: "",
    phone: "",
    email: "",
    password: "",
  });

  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // ==========================================
  // NAVIGATION
  // ==========================================

  const [activePage, setActivePage] =
    useState("dashboard");

  // ==========================================
  // HISTORY
  // ==========================================

  const [history, setHistory] =
    useState([]);

  const [historyLoading, setHistoryLoading] =
    useState(false);

  const [historyError, setHistoryError] =
    useState("");

  // ==========================================
  // SOS
  // ==========================================

  const [sosLoading, setSosLoading] =
    useState(false);

  const [sosMessage, setSosMessage] =
    useState("");

  const [sosError, setSosError] =
    useState("");

  // Prevent repeated automatic SOS
  const automaticSosTriggeredRef =
    useRef(false);

  // ==========================================
  // CONTACTS
  // ==========================================

  const [contacts, setContacts] = useState([]);

  const [contactsLoading, setContactsLoading] =
    useState(false);

  const [showContactForm, setShowContactForm] =
    useState(false);

  const [contactForm, setContactForm] = useState({
    name: "",
    phone: "",
    relation: "",
  });

  const [contactMessage, setContactMessage] =
    useState("");

  const [contactError, setContactError] =
    useState("");

  // ==========================================
  // AI MONITORING
  // ==========================================

  const [monitoring, setMonitoring] = useState({
    active: false,
    voice_detection: false,
    emotion_detection: false,
    camera_detection: false,
  });

  const [monitoringLoading, setMonitoringLoading] =
    useState(false);

  const [monitoringError, setMonitoringError] =
    useState("");

  // ==========================================
  // CAMERA
  // ==========================================

  const videoRef = useRef(null);
  const streamRef = useRef(null);

  const [cameraActive, setCameraActive] =
    useState(false);

  const [cameraLoading, setCameraLoading] =
    useState(false);

  const [cameraResult, setCameraResult] =
    useState(null);

  const [cameraError, setCameraError] =
    useState("");

  // ==========================================
  // MICROPHONE / VOICE
  // ==========================================

  const microphoneStreamRef =
    useRef(null);

  const audioRecorderRef =
    useRef(null);

  const audioChunksRef =
    useRef([]);

  const [microphoneActive, setMicrophoneActive] =
    useState(false);

  const [microphoneError, setMicrophoneError] =
    useState("");

  const [voiceResult, setVoiceResult] =
    useState(null);

  const [emotionResult, setEmotionResult] =
    useState(null);

  const [emergencyWord, setEmergencyWord] =
    useState(null);

  // Periodic pretrained voice/emotion analysis
  const voiceMonitoringIntervalRef =
    useRef(null);

  // ==========================================
  // RISK
  // ==========================================

  const [riskResult, setRiskResult] =
    useState(null);

  // ==========================================
  // USER
  // ==========================================

  const user = JSON.parse(
    localStorage.getItem("user") || "null"
  );

  // ==========================================
  // FORM CHANGE
  // ==========================================

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  // ==========================================
  // LOAD CONTACTS
  // ==========================================

  const loadContacts = async () => {
    setContactsLoading(true);
    setContactError("");

    try {
      const response =
        await getEmergencyContacts();

      setContacts(
        response?.contacts || []
      );
    } catch (err) {
      setContactError(
        err?.message ||
          "Failed to load contacts."
      );
    } finally {
      setContactsLoading(false);
    }
  };

  // ==========================================
  // LOAD MONITORING STATUS
  // ==========================================

  const loadMonitoringStatus = async () => {
    try {
      const response =
        await getMonitoringStatus();

      setMonitoring({
        active: !!response?.active,
        voice_detection:
          !!response?.voice_detection,
        emotion_detection:
          !!response?.emotion_detection,
        camera_detection:
          !!response?.camera_detection,
      });
    } catch (err) {
      setMonitoringError(
        err?.message ||
          "Failed to load monitoring status."
      );
    }
  };

  // ==========================================
  // LOAD HISTORY
  // ==========================================

  const loadHistory = async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await getHistory();
      setHistory(response?.history || []);
    } catch (err) {
      console.error("Failed to load history:", err);
      setHistoryError(
        err?.message ||
          "Failed to load safety history."
      );
      setHistory([]);
    } finally {
      setHistoryLoading(false);
    }
  };

  // ==========================================
  // INITIAL DATA
  // ==========================================

  useEffect(() => {
    if (loggedIn) {
      loadContacts();
      loadMonitoringStatus();
      loadHistory();
    }
  }, [loggedIn]);

  // ==========================================
  // CAMERA STREAM ATTACH
  // ==========================================

  useEffect(() => {
    if (
      cameraActive &&
      videoRef.current &&
      streamRef.current
    ) {
      const video = videoRef.current;

      video.srcObject =
        streamRef.current;

      const handleLoaded = async () => {
        try {
          await video.play();
        } catch (err) {
          console.error(
            "Video play error:",
            err
          );
        }
      };

      video.addEventListener(
        "loadedmetadata",
        handleLoaded
      );

      if (video.readyState >= 1) {
        handleLoaded();
      }

      return () => {
        video.removeEventListener(
          "loadedmetadata",
          handleLoaded
        );
      };
    }
  }, [cameraActive]);

  // ==========================================
  // STOP CAMERA WHEN LEAVING MONITORING
  // ==========================================

  useEffect(() => {
    if (
      activePage !== "monitoring"
    ) {
      stopEmergencyWordDetection();
      stopVoiceEmotionMonitoring();

      if (cameraActive) {
        stopCamera();
      }
    }
  }, [activePage]);

  // ==========================================
  // LOGIN / REGISTER
  // ==========================================

  const handleSubmit = async (e) => {
    e.preventDefault();

    setMessage("");
    setError("");
    setLoading(true);

    try {
      if (mode === "register") {
        await registerUser({
          name: form.name.trim(),
          phone: form.phone.trim(),
          email: form.email.trim(),
          password: form.password,
        });

        setMessage(
          "Registration successful! Please login."
        );

        setMode("login");

        setForm({
          name: "",
          phone: "",
          email: form.email,
          password: "",
        });
      } else {
        await loginUser({
          email: form.email,
          password: form.password,
        });

        setLoggedIn(true);
        setActivePage("dashboard");
      }
    } catch (err) {
      setError(
        err?.message ||
          "Something went wrong."
      );
    } finally {
      setLoading(false);
    }
  };

  // ==========================================
  // LOGOUT
  // ==========================================

  const handleLogout = () => {
    stopEmergencyWordDetection();
    stopVoiceEmotionMonitoring();
    stopCamera();

    logoutUser();

    setLoggedIn(false);
    setActivePage("dashboard");
    setContacts([]);

    setMonitoring({
      active: false,
      voice_detection: false,
      emotion_detection: false,
      camera_detection: false,
    });

    setCameraResult(null);
    setCameraError("");
    setRiskResult(null);

    setMessage("");
    setError("");
    setSosMessage("");
    setSosError("");
    setContactMessage("");
    setContactError("");

    automaticSosTriggeredRef.current =
      false;
  };

  // ==========================================
  // MANUAL SOS
  // ==========================================

  const handleSOS = async () => {
    setSosMessage("");
    setSosError("");

    if (!navigator.geolocation) {
      setSosError(
        "Location is not supported by this browser."
      );
      return;
    }

    setSosLoading(true);

    try {
      const position = await new Promise(
        (resolve, reject) => {
          navigator.geolocation.getCurrentPosition(
            resolve,
            reject,
            {
              enableHighAccuracy: true,
              timeout: 30000,
              maximumAge: 0,
            }
          );
        }
      );

      const latitude =
        position.coords.latitude;

      const longitude =
        position.coords.longitude;

      const response = await triggerSOS({
        latitude,
        longitude,
        message:
          "Emergency! I need help. Please check my location.",
      });

      const sosId =
        response?.sos_id || "N/A";

      const contactCount =
        response?.emergency_contacts?.count ?? 0;

      setSosMessage(
        `SOS activated successfully. SOS ID: ${sosId}. Emergency contacts: ${contactCount}.`
      );

      console.log(
        "Manual SOS response:",
        response
      );
    } catch (err) {
      console.error(
        "Manual SOS failed:",
        err
      );

      if (err?.code === 1) {
        setSosError(
          "Location permission denied. Please allow location access."
        );
      } else if (err?.code === 2) {
        setSosError(
          "Unable to determine your location. Please try again."
        );
      } else if (err?.code === 3) {
        setSosError(
          "Location request timed out. Please try again."
        );
      } else {
        setSosError(
          err?.message ||
            "Failed to activate SOS."
        );
      }
    } finally {
      setSosLoading(false);
    }
  };

  // ==========================================
  // AUTOMATIC SOS
  // ==========================================

  const triggerAutomaticSOS = async (
    riskData
  ) => {
    if (
      automaticSosTriggeredRef.current
    ) {
      console.log(
        "Automatic SOS already triggered."
      );
      return;
    }

    if (!navigator.geolocation) {
      setSosError(
        "Location is not supported by this browser."
      );
      return;
    }

    // Lock immediately to prevent duplicate SOS.
    automaticSosTriggeredRef.current =
      true;

    setSosError("");
    setSosMessage("");

    try {
      const position = await new Promise(
        (resolve, reject) => {
          navigator.geolocation.getCurrentPosition(
            resolve,
            reject,
            {
              enableHighAccuracy: true,
              timeout: 30000,
              maximumAge: 0,
            }
          );
        }
      );

      const latitude =
        position.coords.latitude;

      const longitude =
        position.coords.longitude;

      const response = await triggerSOS({
        latitude,
        longitude,
        message:
          "GuardAI automatically detected a high-risk safety situation. Please check my location immediately.",
      });

      const sosId =
        response?.sos_id || "N/A";

      const contactCount =
        response?.emergency_contacts?.count ?? 0;

      const notificationStatus =
        response?.notification?.status ||
        "unknown";

      const deliveryStatus =
        response?.notification?.delivery ||
        "unknown";

      setSosMessage(
        `Automatic SOS activated. SOS ID: ${sosId}. Emergency contacts: ${contactCount}.`
      );

      console.log(
        "Automatic SOS response:",
        response
      );

      console.log(
        "SOS ID:",
        sosId
      );

      console.log(
        "Emergency contacts:",
        response?.emergency_contacts
      );

      console.log(
        "Notification status:",
        notificationStatus
      );

      console.log(
        "Notification delivery:",
        deliveryStatus
      );

      console.log(
        "Risk which triggered SOS:",
        riskData
      );
    } catch (err) {
      // Allow retry when SOS failed.
      automaticSosTriggeredRef.current =
        false;

      console.error(
        "Automatic SOS failed:",
        err
      );

      if (err?.code === 1) {
        setSosError(
          "Location permission denied. Please allow location access."
        );
      } else if (err?.code === 2) {
        setSosError(
          "Unable to determine your location. Please try again."
        );
      } else if (err?.code === 3) {
        setSosError(
          "Location request timed out. Please try again."
        );
      } else {
        setSosError(
          err?.message ||
            "Automatic SOS could not be activated."
        );
      }
    }
  };

  // ==========================================
  // RISK ANALYSIS
  // ==========================================

  const analyzeCurrentSafety = async (
    imageFile,
    audioFile = null
  ) => {
    try {
      if (
        typeof analyzeSafety !==
        "function"
      ) {
        console.warn(
          "analyzeSafety() is not available in monitoringApi.ts"
        );
        return null;
      }

      const currentHour =
        new Date().getHours();

      // Do not use fake/test distress or emotion
      // values here. The backend will use the
      // actual pretrained detectors when audio
      // is supplied.
      const result = await analyzeSafety({
        audioFile,
        imageFile,
        hour: currentHour,
        alone: false,
        darkArea: false,
      });

      console.log(
        "GuardAI safety analysis:",
        result
      );

      setRiskResult(result);

      if (result?.voice) {
        setVoiceResult(result.voice);
      }

      if (result?.emotion) {
        setEmotionResult(result.emotion);
      }

      const riskLevel =
        String(
          result?.risk?.risk_level || ""
        ).toLowerCase();

      const sosRequired =
        result?.sos_required === true;

      if (
        sosRequired ||
        riskLevel === "high"
      ) {
        console.warn(
          "HIGH RISK DETECTED — triggering automatic SOS."
        );

        await triggerAutomaticSOS(
          result
        );
      }

      return result;
    } catch (err) {
      console.error(
        "Safety analysis failed:",
        err
      );

      setMonitoringError(
        err?.message ||
          "Safety analysis failed."
      );

      return null;
    }
  };

  // ==========================================
  // MICROPHONE AUDIO HELPERS
  // ==========================================

  const audioBlobToWavFile = async (
    audioBlob
  ) => {
    const arrayBuffer =
      await audioBlob.arrayBuffer();

    const audioContext =
      new (
        window.AudioContext ||
        window.webkitAudioContext
      )();

    try {
      const audioBuffer =
        await audioContext.decodeAudioData(
          arrayBuffer
        );

      const targetSampleRate = 16000;

      const offlineContext =
        new OfflineAudioContext(
          1,
          Math.ceil(
            audioBuffer.duration *
              targetSampleRate
          ),
          targetSampleRate
        );

      const source =
        offlineContext.createBufferSource();

      source.buffer = audioBuffer;

      const channelData =
        audioBuffer.numberOfChannels === 1
          ? audioBuffer.getChannelData(0)
          : (() => {
              const left =
                audioBuffer.getChannelData(0);

              const right =
                audioBuffer.getChannelData(1);

              const mono =
                new Float32Array(
                  Math.max(
                    left.length,
                    right.length
                  )
                );

              for (
                let i = 0;
                i < mono.length;
                i++
              ) {
                mono[i] =
                  ((left[i] || 0) +
                    (right[i] || 0)) /
                  2;
              }

              return mono;
            })();

      const tempBuffer =
        offlineContext.createBuffer(
          1,
          channelData.length,
          audioBuffer.sampleRate
        );

      tempBuffer.copyToChannel(
        channelData,
        0
      );

      const tempSource =
        offlineContext.createBufferSource();

      tempSource.buffer = tempBuffer;

      const gain =
        offlineContext.createGain();

      gain.gain.value = 1;

      tempSource.connect(gain);
      gain.connect(
        offlineContext.destination
      );

      tempSource.start(0);

      const renderedBuffer =
        await offlineContext.startRendering();

      const samples =
        renderedBuffer.getChannelData(0);

      const wavBuffer =
        new ArrayBuffer(
          44 + samples.length * 2
        );

      const view =
        new DataView(wavBuffer);

      const writeString = (
        offset,
        string
      ) => {
        for (
          let i = 0;
          i < string.length;
          i++
        ) {
          view.setUint8(
            offset + i,
            string.charCodeAt(i)
          );
        }
      };

      const writeUInt16 = (
        offset,
        value
      ) => {
        view.setUint16(
          offset,
          value,
          true
        );
      };

      const writeUInt32 = (
        offset,
        value
      ) => {
        view.setUint32(
          offset,
          value,
          true
        );
      };

      writeString(0, "RIFF");

      writeUInt32(
        4,
        36 + samples.length * 2
      );

      writeString(8, "WAVE");

      writeString(12, "fmt ");

      writeUInt32(16, 16);

      writeUInt16(20, 1);

      writeUInt16(22, 1);

      writeUInt32(
        24,
        targetSampleRate
      );

      writeUInt32(
        28,
        targetSampleRate * 2
      );

      writeUInt16(32, 2);

      writeUInt16(34, 16);

      writeString(36, "data");

      writeUInt32(
        40,
        samples.length * 2
      );

      let offset = 44;

      for (
        let i = 0;
        i < samples.length;
        i++
      ) {
        const sample =
          Math.max(
            -1,
            Math.min(
              1,
              samples[i]
            )
          );

        const intSample =
          sample < 0
            ? sample * 0x8000
            : sample * 0x7fff;

        view.setInt16(
          offset,
          intSample,
          true
        );

        offset += 2;
      }

      return new File(
        [wavBuffer],
        "guardai_voice.wav",
        {
          type: "audio/wav",
        }
      );
    } finally {
      await audioContext.close();
    }
  };


  const recordVoiceClip = async (
    duration = 3000
  ) => {
    setMicrophoneError("");

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      throw new Error(
        "Microphone is not supported by this browser."
      );
    }

    let stream =
      microphoneStreamRef.current;

    if (!stream) {
      stream =
        await navigator.mediaDevices.getUserMedia(
          {
            audio: {
              channelCount: 1,
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            },
            video: false,
          }
        );

      microphoneStreamRef.current =
        stream;
    }

    setMicrophoneActive(true);

    const mimeTypes = [
      "audio/webm;codecs=opus",
      "audio/webm",
      "audio/ogg;codecs=opus",
    ];

    const supportedMimeType =
      mimeTypes.find(
        (type) =>
          MediaRecorder.isTypeSupported(
            type
          )
      );

    const recorder =
      supportedMimeType
        ? new MediaRecorder(
            stream,
            {
              mimeType:
                supportedMimeType,
            }
          )
        : new MediaRecorder(
            stream
          );

    audioRecorderRef.current =
      recorder;

    audioChunksRef.current = [];

    return new Promise(
      (resolve, reject) => {
        recorder.ondataavailable =
          (event) => {
            if (
              event.data &&
              event.data.size > 0
            ) {
              audioChunksRef.current.push(
                event.data
              );
            }
          };

        recorder.onerror =
          (event) => {
            reject(
              event.error ||
                new Error(
                  "Microphone recording failed."
                )
            );
          };

        recorder.onstop =
          async () => {
            try {
              const blob =
                new Blob(
                  audioChunksRef.current,
                  {
                    type:
                      recorder.mimeType ||
                      "audio/webm",
                  }
                );

              if (!blob.size) {
                throw new Error(
                  "Recorded audio is empty."
                );
              }

              const wavFile =
                await audioBlobToWavFile(
                  blob
                );

              resolve(wavFile);
            } catch (error) {
              reject(error);
            } finally {
              audioChunksRef.current =
                [];
              audioRecorderRef.current =
                null;
            }
          };

        recorder.start();

        setTimeout(() => {
          if (
            recorder.state ===
            "recording"
          ) {
            recorder.stop();
          }
        }, duration);
      }
    );
  };


  const stopMicrophone = () => {
    if (
      audioRecorderRef.current &&
      audioRecorderRef.current.state !==
        "inactive"
    ) {
      audioRecorderRef.current.stop();
    }

    if (
      microphoneStreamRef.current
    ) {
      microphoneStreamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      microphoneStreamRef.current =
        null;
    }

    setMicrophoneActive(false);
  };

  // ==========================================
  // PRETRAINED VOICE + EMOTION MONITORING
  // ==========================================
const startVoiceEmotionMonitoring =
  async () => {

    if (
      voiceMonitoringIntervalRef.current
    ) {
      return;
    }

    const runVoiceAnalysis =
      async () => {

        try {

          // ----------------------------------
          // Temporarily stop browser speech
          // recognition so MediaRecorder can
          // safely use the microphone.
          // ----------------------------------

          if (
            speechRecognitionRef.current
          ) {
            stopEmergencyWordDetection();

            // Give the browser a moment
            // to release the speech-recognition
            // microphone session.
            await new Promise(
              (resolve) =>
                setTimeout(
                  resolve,
                  300
                )
            );
          }

          // ----------------------------------
          // Record 3-second microphone clip
          // ----------------------------------

          const audioFile =
            await recordVoiceClip(
              3000
            );

          // ----------------------------------
          // Pretrained Voice + Emotion
          // ----------------------------------

          const result =
            await analyzeCurrentSafety(
              null,
              audioFile
            );

          console.log(
            "Pretrained voice/emotion result:",
            result
          );

        } catch (error) {

          console.warn(
            "Voice/emotion monitoring error:",
            error
          );

        } finally {

          // ----------------------------------
          // Restart emergency-word detection
          // after MediaRecorder releases
          // the microphone.
          // ----------------------------------

          if (
            !speechRecognitionRef.current
          ) {
            startEmergencyWordDetection();
          }
        }
      };

    // Run once immediately.
    await runVoiceAnalysis();

    // Then capture a short clip periodically.
    voiceMonitoringIntervalRef.current =
      window.setInterval(
        runVoiceAnalysis,
        8000
      );
  };

  const stopVoiceEmotionMonitoring =
    () => {
      if (
        voiceMonitoringIntervalRef.current
      ) {
        window.clearInterval(
          voiceMonitoringIntervalRef.current
        );

        voiceMonitoringIntervalRef.current =
          null;
      }

      stopMicrophone();
    };

  const speechRecognitionRef =
  useRef(null);

const emergencyKeywords = [
  "help",
  "help me",
  "save me",
  "emergency",
  "bachao",
  "bachaao",
  "mujhe bachao",
  "madad",
  "madad karo",
  "police",
];

const findEmergencyKeyword = (
  text
) => {
  const normalized =
    String(text || "")
      .toLowerCase()
      .replace(/[.,!?;:]/g, " ")
      .replace(/\s+/g, " ")
      .trim();

  // Normalize common speech-recognition variations
  // Example:
  // bachaao -> bachao
  // bachaoo -> bachao
  // bacchao -> bachao
  const normalizedSpeech =
    normalized
      .replace(/aa+/g, "a")
      .replace(/ee+/g, "e")
      .replace(/ii+/g, "i")
      .replace(/oo+/g, "o")
      .replace(/uu+/g, "u")
      .replace(/(.)\1+/g, "$1");

  const keyword =
    emergencyKeywords.find(
      (keyword) => {
        const normalizedKeyword =
          keyword
            .toLowerCase()
            .replace(/aa+/g, "a")
            .replace(/ee+/g, "e")
            .replace(/ii+/g, "i")
            .replace(/oo+/g, "o")
            .replace(/uu+/g, "u")
            .replace(/(.)\1+/g, "$1");

        return (
          normalizedSpeech.includes(
            normalizedKeyword
          )
        );
      }
    );

  return keyword || null;
};

 const startEmergencyWordDetection =
  () => {
    const SpeechRecognition =
      window.SpeechRecognition ||
      window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn(
        "Speech Recognition not supported."
      );
      return;
    }

    const recognition =
      new SpeechRecognition();

    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-IN";

    recognition.onresult =
      async (event) => {
        let text = "";

        for (
          let i = event.resultIndex;
          i < event.results.length;
          i++
        ) {
          text +=
            " " +
            event.results[i][0]
              .transcript;
        }

        const keyword =
          findEmergencyKeyword(text);

        if (!keyword) {
          return;
        }

        console.warn(
          "Emergency keyword:",
          keyword
        );

        setEmergencyWord(
          keyword
        );

        if (
          automaticSosTriggeredRef.current
        ) {
          return;
        }

        await triggerAutomaticSOS({
          trigger:
            "emergency_keyword",
          keyword,
        });
      };

    recognition.onerror =
      (event) => {
        console.warn(
          "Speech recognition:",
          event.error
        );
      };

    recognition.onend = () => {
      if (
        speechRecognitionRef.current ===
        recognition
      ) {
        try {
          recognition.start();
        } catch {}
      }
    };

    speechRecognitionRef.current =
      recognition;

    try {
      recognition.start();
    } catch {}
  };

const stopEmergencyWordDetection =
  () => {
    if (
      speechRecognitionRef.current
    ) {
      try {
        speechRecognitionRef.current.stop();
      } catch {}

      speechRecognitionRef.current =
        null;
    }

    setEmergencyWord(null);
  };

  // ==========================================
  // START MONITORING
  // ==========================================

  const handleStartMonitoring =
    async () => {
      setMonitoringLoading(true);
      setMonitoringError("");
      setSosMessage("");
      setSosError("");
      setRiskResult(null);

      automaticSosTriggeredRef.current =
        false;

      try {
        const response =
          await startMonitoring({
            voice_detection: true,
            emotion_detection: true,
            camera_detection: true,
          });

        if (response?.monitoring) {
          setMonitoring(
            response.monitoring
          );
        } else {
          await loadMonitoringStatus();
        }

        // AI monitoring includes camera monitoring.
        // Start the browser camera automatically.
        await startCamera();

        // Start emergency-word detection.
        startEmergencyWordDetection();

        // Start pretrained voice + emotion analysis.
        // This uses short microphone clips and the
        // already-loaded backend models.
        await startVoiceEmotionMonitoring();

      } catch (err) {
        setMonitoringError(
          err?.message ||
            "Failed to start monitoring."
        );
      } finally {
        setMonitoringLoading(false);
      }
    };

  // ==========================================
  // STOP MONITORING
  // ==========================================

  const handleStopMonitoring =
    async () => {
      setMonitoringLoading(true);
      setMonitoringError("");

      try {
        const response =
          await stopMonitoring();

        if (response?.monitoring) {
          setMonitoring(
            response.monitoring
          );
        } else {
          await loadMonitoringStatus();
        }

        stopEmergencyWordDetection();
        stopVoiceEmotionMonitoring();
        stopCamera();

        setVoiceResult(null);
        setEmotionResult(null);
        setRiskResult(null);
        setSosMessage("");
        setSosError("");

        automaticSosTriggeredRef.current =
          false;
      } catch (err) {
        setMonitoringError(
          err?.message ||
            "Failed to stop monitoring."
        );
      } finally {
        setMonitoringLoading(false);
      }
    };

  // ==========================================
  // START CAMERA
  // ==========================================

  const startCamera = async () => {
    setCameraError("");
    setCameraResult(null);
    setRiskResult(null);

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {
      setCameraError(
        "Camera is not supported by this browser."
      );

      return;
    }

    try {
      if (streamRef.current) {
        streamRef.current
          .getTracks()
          .forEach((track) => {
            track.stop();
          });

        streamRef.current = null;
      }

      const stream =
        await navigator.mediaDevices.getUserMedia(
          {
            video: {
              facingMode: "user",
            },
            audio: false,
          }
        );

      streamRef.current = stream;

      setCameraActive(true);
    } catch (err) {
      console.error(
        "Camera error:",
        err
      );

      setCameraError(
        "Camera access denied. Please allow camera permission in your browser."
      );

      setCameraActive(false);
    }
  };

  // ==========================================
  // STOP CAMERA
  // ==========================================

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current
        .getTracks()
        .forEach((track) => {
          track.stop();
        });

      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.srcObject = null;
    }

    setCameraActive(false);
  };

  // ==========================================
  // CAPTURE CAMERA IMAGE
  // ==========================================

  const captureCameraImage =
    async () => {
      const video = videoRef.current;

      if (!video) {
        setCameraError(
          "Camera preview is not available."
        );

        return;
      }

      if (
        video.readyState < 2 ||
        video.videoWidth === 0 ||
        video.videoHeight === 0
      ) {
        setCameraError(
          "Camera is still starting. Please wait 1-2 seconds and try again."
        );

        return;
      }

      setCameraLoading(true);
      setCameraError("");
      setCameraResult(null);
      setRiskResult(null);

      try {
        const canvas =
          document.createElement(
            "canvas"
          );

        canvas.width =
          video.videoWidth;

        canvas.height =
          video.videoHeight;

        const context =
          canvas.getContext("2d");

        if (!context) {
          throw new Error(
            "Unable to capture camera image."
          );
        }

        context.drawImage(
          video,
          0,
          0,
          canvas.width,
          canvas.height
        );

        const blob =
          await new Promise(
            (resolve) => {
              canvas.toBlob(
                resolve,
                "image/jpeg",
                0.85
              );
            }
          );

        if (!blob) {
          throw new Error(
            "Failed to create camera image."
          );
        }

        const file = new File(
          [blob],
          "camera_capture.jpg",
          {
            type: "image/jpeg",
          }
        );

        console.log(
          "Camera image captured:",
          file.size,
          "bytes"
        );

        // =====================================
        // CAMERA AI
        // =====================================

        const result =
          await detectCamera(file);

        console.log(
          "Camera AI result:",
          result
        );

        setCameraResult(result);

        // =====================================
        // RISK ANALYSIS
        // =====================================

        await analyzeCurrentSafety(
          file
        );
      } catch (err) {
        console.error(
          "Camera analysis error:",
          err
        );

        setCameraError(
          err?.message ||
            "Camera analysis failed."
        );
      } finally {
        setCameraLoading(false);
      }
    };

  // ==========================================
  // ADD CONTACT
  // ==========================================

  const handleAddContact = async (e) => {
    e.preventDefault();

    setContactMessage("");
    setContactError("");

    try {
      await addEmergencyContact({
        name: contactForm.name.trim(),
        phone: contactForm.phone.trim(),
        relation:
          contactForm.relation.trim() ||
          undefined,
      });

      setContactMessage(
        "Emergency contact added successfully."
      );

      setContactForm({
        name: "",
        phone: "",
        relation: "",
      });

      setShowContactForm(false);

      await loadContacts();
    } catch (err) {
      setContactError(
        err?.message ||
          "Failed to add contact."
      );
    }
  };

  // ==========================================
  // DELETE CONTACT
  // ==========================================

  const handleDeleteContact = async (
    contactId
  ) => {
    if (!contactId) return;

    const confirmed =
      window.confirm(
        "Are you sure you want to delete this emergency contact?"
      );

    if (!confirmed) return;

    setContactMessage("");
    setContactError("");

    try {
      await deleteEmergencyContact(
        contactId
      );

      setContactMessage(
        "Emergency contact deleted successfully."
      );

      await loadContacts();
    } catch (err) {
      setContactError(
        err?.message ||
          "Failed to delete contact."
      );
    }
  };

  // ==========================================
  // LOGIN PAGE
  // ==========================================

  if (!loggedIn) {
    return (
      <div className="login-page">
        <div className="login-card">

          <div className="login-logo">
            🛡️
          </div>

          <h1>GuardAI</h1>

          <p className="login-subtitle">
            AI-Powered Women Safety
          </p>

          <h2>
            {mode === "login"
              ? "Welcome Back"
              : "Create Account"}
          </h2>

          <p className="login-description">
            {mode === "login"
              ? "Your safety companion is always with you."
              : "Create your GuardAI account to stay protected."}
          </p>

          <form
            onSubmit={handleSubmit}
          >
            {mode === "register" && (
              <input
                type="text"
                name="name"
                placeholder="Full Name"
                value={form.name}
                onChange={handleChange}
                required
              />
            )}

            {mode === "register" && (
              <input
                type="tel"
                name="phone"
                placeholder="Mobile Number"
                value={form.phone}
                onChange={handleChange}
                required
                inputMode="numeric"
                pattern="[0-9]{10,15}"
                minLength={10}
                maxLength={15}
                title="Please enter a valid 10-15 digit mobile number"
              />
            )}

            <input
              type="email"
              name="email"
              placeholder="Email Address"
              value={form.email}
              onChange={handleChange}
              required
            />

            <input
              type="password"
              name="password"
              placeholder="Password"
              value={form.password}
              onChange={handleChange}
              required
            />

            <button
              className="primary-button"
              type="submit"
              disabled={loading}
            >
              {loading
                ? "Please wait..."
                : mode === "login"
                ? "Login"
                : "Create Account"}
            </button>
          </form>

          {message && (
            <p className="success-message">
              ✓ {message}
            </p>
          )}

          {error && (
            <p className="error-message">
              ⚠ {error}
            </p>
          )}

          <button
            className="switch-button"
            type="button"
            onClick={() => {
              setMode(
                mode === "login"
                  ? "register"
                  : "login"
              );

              setMessage("");
              setError("");
            }}
          >
            {mode === "login"
              ? "Don't have an account? Create one"
              : "Already have an account? Login"}
          </button>

        </div>
      </div>
    );
  }

  // ==========================================
  // SIDEBAR
  // ==========================================

  const renderSidebar = () => (
    <aside className="sidebar">

      <div className="logo">
        <h1>🛡️ GuardAI</h1>
        <p>
          AI-Powered Women Safety
        </p>
      </div>

      <div className="nav-menu">

        <button
          className={`nav-item ${
            activePage === "dashboard"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("dashboard")
          }
        >
          🏠 <span>Dashboard</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "sos"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("sos")
          }
        >
          🚨 <span>SOS Emergency</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "contacts"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("contacts")
          }
        >
          👥 <span>Emergency Contacts</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "monitoring"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("monitoring")
          }
        >
          🤖 <span>AI Monitoring</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "location"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("location")
          }
        >
          📍 <span>Location Tracker</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "safety"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("safety")
          }
        >
          🛡️ <span>Safety Tips</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "history"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("history")
          }
        >
          🕘 <span>History</span>
        </button>

        <button
          className={`nav-item ${
            activePage === "profile"
              ? "active"
              : ""
          }`}
          onClick={() =>
            setActivePage("profile")
          }
        >
          👤 <span>Profile</span>
        </button>

        <button
          className="nav-item"
          onClick={handleLogout}
        >
          🚪 <span>Logout</span>
        </button>

      </div>

      <div className="sidebar-bottom">
        <h4>
          💜 You are not alone.
        </h4>

        <p>
          GuardAI is always here to help
          you stay safe.
        </p>
      </div>

    </aside>
  );

  // ==========================================
  // HEADER
  // ==========================================

  const renderHeader = () => (
    <header className="header">

      <div>
        <h2>
          Welcome back,{" "}
          {user?.name || "there"}! 👋
        </h2>

        <p>
          Stay aware. Stay safe.
          We've got your back.
        </p>
      </div>

      <div className="profile">

        <div className="profile-avatar">
          👩
        </div>

        <div>
          <div className="profile-name">
            {user?.name || "User"}
          </div>

          <div className="profile-email">
            {user?.email || ""}
          </div>
        </div>

      </div>

    </header>
  );

  // ==========================================
  // DASHBOARD
  // ==========================================

  const renderDashboard = () => (
    <>
      <section className="status-grid">

        <div className="status-card">
          <div className="status-title">
            Safety Status
          </div>

          <div className="status-value safe">
            {riskResult?.risk?.risk_level ||
              "Safe"}
          </div>

          <div className="status-small">
            {riskResult?.risk?.risk_level
              ? "AI risk analysis"
              : "All systems normal"}
          </div>

          <span className="status-badge">
            Updated just now
          </span>
        </div>

        <div className="status-card">
          <div className="status-title">
            Risk Level
          </div>

          <div className="status-value low">
            {riskResult?.risk?.risk_level ||
              "Low"}
          </div>

          <div className="status-small">
            AI safety assessment
          </div>

          <span className="status-badge">
            Updated just now
          </span>
        </div>

        <div className="status-card">

          <div className="status-title">
            AI Monitoring
          </div>

          <div
            className={`status-value ${
              monitoring.active
                ? "active"
                : "low"
            }`}
          >
            {monitoring.active
              ? "Active"
              : "Off"}
          </div>

          <div className="status-small">
            {monitoring.active
              ? "Monitoring in background"
              : "Monitoring is off"}
          </div>

          <span className="status-badge">
            {monitoring.active
              ? "Active"
              : "Inactive"}
          </span>

        </div>

        <div className="status-card">

          <div className="status-title">
            Trusted Contacts
          </div>

          <div className="status-value contacts-count">
            {contacts.length}
          </div>

          <div className="status-small">
            Contacts added
          </div>

          <span className="status-badge">
            Manage
          </span>

        </div>

      </section>

      <section className="content-grid">

        <div className="card sos-card">

          <div className="card-header">
            <h3>
              🚨 Quick SOS
            </h3>
          </div>

          <div className="sos-content">

            <div className="sos-info">
              <p>
                Press the button in an
                emergency. Your trusted
                contacts can be alerted.
              </p>
            </div>

            <button
              className="sos-button"
              onClick={handleSOS}
              disabled={sosLoading}
            >
              {sosLoading
                ? "..."
                : "SOS"}
            </button>

          </div>

          <div className="sos-note">
            🛡️ Your emergency alert can
            include your live location.
          </div>

          {sosMessage && (
            <p className="success-message">
              ✓ {sosMessage}
            </p>
          )}

          {sosError && (
            <p className="error-message">
              ⚠ {sosError}
            </p>
          )}

        </div>

        <div className="card">

          <div className="card-header">
            <h3>
              📍 Current Location
            </h3>

            <span className="status-badge">
              🟢 Live
            </span>
          </div>

          <div className="map">
            <div className="map-pin">
              📍
            </div>
          </div>

          <div className="location-text">
            Your location is available
            when SOS is triggered.
          </div>

          <button
            className="location-button"
            onClick={() =>
              setActivePage("location")
            }
          >
            Open Location
          </button>

        </div>

        <div className="card">

          <div className="card-header">
            <h3>
              Emergency Contacts
            </h3>

            <button
              className="view-all"
              onClick={() =>
                setActivePage("contacts")
              }
            >
              View All
            </button>
          </div>

          {contacts.length === 0 ? (
            <p className="item-subtext">
              No emergency contacts added.
            </p>
          ) : (
            contacts
              .slice(0, 3)
              .map((contact) => {
                const id =
                  contact._id ||
                  contact.id;

                return (
                  <div
                    className="contact"
                    key={id}
                  >

                    <div className="contact-avatar">
                      👤
                    </div>

                    <div className="contact-info">

                      <div className="contact-name">
                        {contact.name}
                      </div>

                      <div className="contact-number">
                        {contact.phone}
                      </div>

                    </div>

                    <button
                      className="call-button"
                      onClick={() =>
                        (window.location.href =
                          `tel:${contact.phone}`)
                      }
                    >
                      📞
                    </button>

                  </div>
                );
              })
          )}

          <button
            className="add-contact"
            onClick={() => {
              setActivePage("contacts");
              setShowContactForm(true);
            }}
          >
            + Add Emergency Contact
          </button>

        </div>

      </section>

      <section className="lower-grid">

        <div className="card">

          <div className="card-header">
            <h3>
              Recent Activity
            </h3>
          </div>

          <div className="activity-item">

            <div className="activity-icon">
              🟢
            </div>

            <div>

              <div className="item-text">
                GuardAI session active
              </div>

              <div className="item-subtext">
                Current session
              </div>

            </div>

          </div>

          <div className="activity-item">

            <div className="activity-icon">
              🤖
            </div>

            <div>

              <div className="item-text">
                AI monitoring:
                {" "}
                {monitoring.active
                  ? "Active"
                  : "Inactive"}
              </div>

              <div className="item-subtext">
                System status
              </div>

            </div>

          </div>

        </div>

        <div className="card">

          <div className="card-header">
            <h3>
              Safety Tips
            </h3>
          </div>

          <div className="tip-item">

            <div className="tip-icon">
              🛡️
            </div>

            <div className="item-text">
              Stay in well-lit areas at night.
            </div>

          </div>

          <div className="tip-item">

            <div className="tip-icon">
              📍
            </div>

            <div className="item-text">
              Share your location with trusted
              contacts.
            </div>

          </div>

          <div className="tip-item">

            <div className="tip-icon">
              📱
            </div>

            <div className="item-text">
              Keep emergency numbers accessible.
            </div>

          </div>

        </div>

        <div className="card help-card">

          <div className="card-header">
            <h3>
              Need Help?
            </h3>
          </div>

          <p>
            GuardAI is here whenever you
            need assistance.
          </p>

          <button
            className="help-button"
            onClick={() =>
              setActivePage("sos")
            }
          >
            Emergency Help
          </button>

        </div>

      </section>
    </>
  );

  // ==========================================
  // SOS PAGE
  // ==========================================

  const renderSOS = () => (
    <div className="page-content">

      <div className="page-title">

        <h1>
          🚨 SOS Emergency
        </h1>

        <p>
          Get immediate assistance when
          you need it.
        </p>

      </div>

      <div className="card sos-page-card">

        <h2>
          Are you in an emergency?
        </h2>

        <p>
          Press the SOS button to send
          your current location to the
          GuardAI emergency system.
        </p>

        <button
          className="sos-button large-sos"
          onClick={handleSOS}
          disabled={sosLoading}
        >
          {sosLoading
            ? "Sending..."
            : "SOS"}
        </button>

        {sosMessage && (
          <p className="success-message">
            ✓ {sosMessage}
          </p>
        )}

        {sosError && (
          <p className="error-message">
            ⚠ {sosError}
          </p>
        )}

      </div>

    </div>
  );

  // ==========================================
  // CONTACTS PAGE
  // ==========================================

  const renderContacts = () => (
    <div className="page-content">

      <div className="page-title">

        <h1>
          👥 Emergency Contacts
        </h1>

        <p>
          People you trust in an emergency.
        </p>

      </div>

      <div className="card">

        <div className="card-header">

          <h3>
            Your Trusted Contacts
          </h3>

          <span className="status-badge">
            {contacts.length} Contacts
          </span>

        </div>

        {contactsLoading && (
          <p className="item-subtext">
            Loading contacts...
          </p>
        )}

        {!contactsLoading &&
          contacts.length === 0 && (
            <div className="empty-state">

              <div className="empty-icon">
                👥
              </div>

              <h3>
                No emergency contacts
              </h3>

              <p>
                Add someone you trust so
                they can help during an
                emergency.
              </p>

            </div>
          )}

        {!contactsLoading &&
          contacts.map((contact) => {

            const id =
              contact._id ||
              contact.id;

            return (
              <div
                className="contact contact-large"
                key={id}
              >

                <div className="contact-avatar">
                  👤
                </div>

                <div className="contact-info">

                  <div className="contact-name">
                    {contact.name}
                  </div>

                  <div className="contact-number">
                    {contact.phone}
                  </div>

                  {contact.relation && (
                    <div className="contact-number">
                      {contact.relation}
                    </div>
                  )}

                </div>

                <button
                  className="call-button"
                  onClick={() =>
                    (window.location.href =
                      `tel:${contact.phone}`)
                  }
                >
                  📞
                </button>

                <button
                  className="call-button"
                  onClick={() =>
                    handleDeleteContact(id)
                  }
                >
                  🗑️
                </button>

              </div>
            );
          })}

        <button
          className="add-contact"
          onClick={() =>
            setShowContactForm(
              !showContactForm
            )
          }
        >
          {showContactForm
            ? "✕ Cancel"
            : "+ Add Emergency Contact"}
        </button>

        {showContactForm && (
          <form
            onSubmit={handleAddContact}
            className="contact-form"
          >

            <h3>
              Add Trusted Contact
            </h3>

            <input
              type="text"
              placeholder="Full Name"
              value={contactForm.name}
              onChange={(e) =>
                setContactForm({
                  ...contactForm,
                  name: e.target.value,
                })
              }
              required
            />

            <input
              type="tel"
              placeholder="Phone Number"
              value={contactForm.phone}
              onChange={(e) =>
                setContactForm({
                  ...contactForm,
                  phone: e.target.value,
                })
              }
              required
            />

            <input
              type="text"
              placeholder="Relation (Mom, Dad, Friend...)"
              value={contactForm.relation}
              onChange={(e) =>
                setContactForm({
                  ...contactForm,
                  relation: e.target.value,
                })
              }
            />

            <button
              type="submit"
              className="primary-button"
            >
              Save Contact
            </button>

          </form>
        )}

        {contactMessage && (
          <p className="success-message">
            ✓ {contactMessage}
          </p>
        )}

        {contactError && (
          <p className="error-message">
            ⚠ {contactError}
          </p>
        )}

      </div>

    </div>
  );

  // ==========================================
  // AI MONITORING
  // ==========================================

  const renderMonitoring = () => (
    <div className="page-content">

      <div className="page-title">

        <h1>
          🤖 AI Monitoring
        </h1>

        <p>
          GuardAI continuously monitors
          your safety signals.
        </p>

      </div>

      {/* MONITORING STATUS */}

      <div className="card">

        <div className="card-header">

          <h3>
            AI Safety Protection
          </h3>

          <span className="status-badge">
            {monitoring.active
              ? "🟢 Active"
              : "⚪ Inactive"}
          </span>

        </div>

        <div
          className={`status-value ${
            monitoring.active
              ? "active"
              : "low"
          }`}
        >
          {monitoring.active
            ? "Monitoring Active"
            : "Monitoring Off"}
        </div>

        <p>
          {monitoring.active
            ? "GuardAI is monitoring your safety signals."
            : "Start monitoring to activate AI safety protection."}
        </p>

        <div className="monitoring-options">

          <div className="status-card">

            <div className="status-title">
              🎤 Voice Detection
            </div>

            <div
              className={`status-value ${
                monitoring.voice_detection
                  ? "active"
                  : "low"
              }`}
            >
              {monitoring.voice_detection
                ? "Enabled"
                : "Disabled"}
            </div>

          </div>

          <div className="status-card">

            <div className="status-title">
              😊 Emotion Detection
            </div>

            <div
              className={`status-value ${
                monitoring.emotion_detection
                  ? "active"
                  : "low"
              }`}
            >
              {monitoring.emotion_detection
                ? "Enabled"
                : "Disabled"}
            </div>

          </div>

          <div className="status-card">

            <div className="status-title">
              📷 Camera Detection
            </div>

            <div
              className={`status-value ${
                monitoring.camera_detection
                  ? "active"
                  : "low"
              }`}
            >
              {cameraActive
                ? "Enabled"
                : "Disabled"}
            </div>

          </div>

        </div>

        <div
          style={{
            marginTop: "25px",
          }}
        >

          {!monitoring.active ? (
            <button
              className="primary-button"
              onClick={
                handleStartMonitoring
              }
              disabled={monitoringLoading}
            >
              {monitoringLoading
                ? "Starting..."
                : "▶ Start Monitoring"}
            </button>
          ) : (
            <button
              className="help-button"
              onClick={
                handleStopMonitoring
              }
              disabled={monitoringLoading}
            >
              {monitoringLoading
                ? "Stopping..."
                : "⏹ Stop Monitoring"}
            </button>
          )}

        </div>

        {monitoringError && (
          <p className="error-message">
            ⚠ {monitoringError}
          </p>
        )}

      </div>

      {/* CAMERA */}

      <div
        className="card camera-card"
        style={{
          marginTop: "20px",
        }}
      >

        <div className="card-header">

          <h3>
            📷 Camera Safety Detection
          </h3>

          <span className="status-badge">
            {cameraActive
              ? "🟢 Camera On"
              : "⚪ Camera Off"}
          </span>

        </div>

        <p>
          Camera analysis can help GuardAI
          detect people and possible safety
          situations.
        </p>

        {!cameraActive ? (
          <button
            className="primary-button"
            onClick={startCamera}
          >
            📷 Enable Camera
          </button>
        ) : (
          <>

            <div
              style={{
                width: "100%",
                maxWidth: "650px",
                margin: "20px auto",
                borderRadius: "18px",
                overflow: "hidden",
                background: "#111",
              }}
            >

              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: "100%",
                  display: "block",
                  minHeight: "300px",
                  objectFit: "cover",
                }}
              />

            </div>

            <div
              style={{
                display: "flex",
                gap: "12px",
                flexWrap: "wrap",
                marginTop: "15px",
              }}
            >

              <button
                className="primary-button"
                onClick={
                  captureCameraImage
                }
                disabled={
                  cameraLoading
                }
              >
                {cameraLoading
                  ? "🤖 Analyzing..."
                  : "📸 Capture & Analyze"}
              </button>

              <button
                className="help-button"
                onClick={stopCamera}
                disabled={
                  cameraLoading
                }
              >
                ⏹ Stop Camera
              </button>

            </div>

          </>
        )}

        {cameraError && (
          <p className="error-message">
            ⚠ {cameraError}
          </p>
        )}

        {cameraResult && (
          <div
            style={{
              marginTop: "20px",
              padding: "18px",
              borderRadius: "14px",
              background: "#f7f7fb",
            }}
          >

            <h4>
              🤖 AI Camera Result
            </h4>

            <pre
              style={{
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                fontSize: "13px",
              }}
            >
              {JSON.stringify(
                cameraResult,
                null,
                2
              )}
            </pre>

          </div>
        )}

        {riskResult && (
          <div
            style={{
              marginTop: "20px",
              padding: "18px",
              borderRadius: "14px",
              background: "#f7f7fb",
            }}
          >

            <h4>
              🛡️ GuardAI Risk Analysis
            </h4>

            <pre
              style={{
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                fontSize: "13px",
              }}
            >
              {JSON.stringify(
                riskResult,
                null,
                2
              )}
            </pre>

          </div>
        )}

      </div>

    </div>
  );

  // ==========================================
  // LOCATION
  // ==========================================

  const renderLocation = () => (
    <div className="page-content">

      <div className="page-title">

        <h1>
          📍 Location Tracker
        </h1>

        <p>
          Your current safety location.
        </p>

      </div>

      <div className="card">

        <div className="map large-map">
          <div className="map-pin">
            📍
          </div>
        </div>

        <h3>
          Location Services
        </h3>

        <p>
          Your location is requested
          when an emergency SOS is
          triggered.
        </p>

        <button
          className="primary-button"
          onClick={handleSOS}
        >
          Test Location
        </button>

      </div>

    </div>
  );

  // ==========================================
  // SAFETY
  // ==========================================

  const renderSafety = () => (
    <div className="page-content">

      <div className="page-title">

        <h1>
          🛡️ Safety Tips
        </h1>

        <p>
          Simple habits that can help
          you stay safer.
        </p>

      </div>

      <div className="lower-grid">

        <div className="card">
          <h3>
            🌙 At Night
          </h3>

          <p>
            Stay in well-lit and populated
            areas whenever possible.
          </p>
        </div>

        <div className="card">
          <h3>
            📍 Location
          </h3>

          <p>
            Share your trip or location
            with someone you trust.
          </p>
        </div>

        <div className="card">
          <h3>
            📱 Emergency
          </h3>

          <p>
            Keep emergency contacts
            accessible on your phone.
          </p>
        </div>

      </div>

    </div>
  );

  // ==========================================
  // HISTORY
  // ==========================================

  const renderHistory = () => (
    <div className="page-content">

      <div className="page-title">
        <h1>
          🕘 Safety History
        </h1>

        <p>
          Your recent GuardAI activity.
        </p>
      </div>

      {historyLoading && (
        <div className="card">
          <div className="item-text">
            Loading safety history...
          </div>
        </div>
      )}

      {historyError && (
        <div className="card">
          <div className="item-text">
            Unable to load history
          </div>

          <div className="item-subtext">
            {historyError}
          </div>
        </div>
      )}

      {!historyLoading &&
        !historyError &&
        history.length === 0 && (
          <div className="card">
            <div className="activity-item">
              <div className="activity-icon">
                📝
              </div>

              <div>
                <div className="item-text">
                  No safety activity yet
                </div>

                <div className="item-subtext">
                  Your SOS activity will appear here.
                </div>
              </div>
            </div>
          </div>
        )}

      {!historyLoading &&
        !historyError &&
        history.length > 0 && (
          <div className="card">
            {history.map((item) => {
              const createdAt = item.created_at
                ? new Date(item.created_at).toLocaleString()
                : "Unknown time";

              const isResolved =
                item.status?.toLowerCase() === "resolved";

              return (
                <div
                  className="activity-item"
                  key={item.id}
                >
                  <div className="activity-icon">
                    {isResolved ? "✅" : "🚨"}
                  </div>

                  <div>
                    <div className="item-text">
                      {item.type || "SOS"} -{" "}
                      {isResolved ? "Resolved" : "Active"}
                    </div>

                    <div className="item-subtext">
                      {createdAt}
                    </div>

                    {item.message && (
                      <div className="item-subtext">
                        {item.message}
                      </div>
                    )}

                    {item.latitude !== null &&
                      item.longitude !== null && (
                        <div className="item-subtext">
                          📍 Location: {item.latitude},{" "}
                          {item.longitude}
                        </div>
                      )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

    </div>
  );

  // ==========================================
  // PROFILE
  // ==========================================

  const renderProfile = () => (
    <div className="page-content">

      <div className="page-title">

        <h1>
          👤 My Profile
        </h1>

        <p>
          Manage your GuardAI account.
        </p>

      </div>

      <div className="card profile-card">

        <div className="profile-avatar large-avatar">
          👩
        </div>

        <h2>
          {user?.name || "User"}
        </h2>

        <p>
          {user?.email || ""}
        </p>

        <span className="status-badge">
          GuardAI Member
        </span>

      </div>

    </div>
  );

  // ==========================================
  // PAGE SELECTOR
  // ==========================================

  const renderPage = () => {
    switch (activePage) {

      case "sos":
        return renderSOS();

      case "contacts":
        return renderContacts();

      case "monitoring":
        return renderMonitoring();

      case "location":
        return renderLocation();

      case "safety":
        return renderSafety();

      case "history":
        return renderHistory();

      case "profile":
        return renderProfile();

      case "dashboard":
      default:
        return renderDashboard();
    }
  };

  // ==========================================
  // MAIN
  // ==========================================

  return (
    <div className="dashboard">

      {renderSidebar()}

      <main className="main">

        {renderHeader()}

        {renderPage()}

        <div className="footer-banner">

          <div
            style={{
              fontSize: "30px",
            }}
          >
            🛡️
          </div>

          <div>

            <strong>
              GuardAI is active and protecting
              you.
            </strong>

            <p>
              Stay safe, stay fearless! 💜
            </p>

          </div>

        </div>

      </main>

    </div>
  );
}

export default App;