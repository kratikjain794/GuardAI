import apiRequest from "./apiClient";

// ==========================================
// TYPES
// ==========================================

export interface MonitoringSettings {
  voice_detection: boolean;
  emotion_detection: boolean;
  camera_detection: boolean;
}

export interface MonitoringState {
  active: boolean;
  voice_detection: boolean;
  emotion_detection: boolean;
  camera_detection: boolean;
}

// ==========================================
// GET MONITORING STATUS
// ==========================================

export async function getMonitoringStatus() {
  return apiRequest("/monitoring/status");
}

// ==========================================
// START MONITORING
// ==========================================

export async function startMonitoring(
  settings: MonitoringSettings
) {
  return apiRequest("/monitoring/start", {
    method: "POST",
    body: JSON.stringify(settings),
  });
}

// ==========================================
// STOP MONITORING
// ==========================================

export async function stopMonitoring() {
  return apiRequest("/monitoring/stop", {
    method: "POST",
  });
}

// ==========================================
// EMOTION DETECTION
// ==========================================

export async function detectEmotion(
  audioFile: File
) {
  const formData = new FormData();

  formData.append("audio", audioFile);

  return apiRequest("/monitoring/emotion", {
    method: "POST",
    body: formData,
  });
}

// ==========================================
// DISTRESS DETECTION
// ==========================================

export async function detectDistress(
  audioFile: File
) {
  const formData = new FormData();

  formData.append("audio", audioFile);

  return apiRequest("/monitoring/distress", {
    method: "POST",
    body: formData,
  });
}

// ==========================================
// CAMERA DETECTION
// ==========================================

export async function detectCamera(
  imageFile: File
) {
  const formData = new FormData();

  formData.append("image", imageFile);

  return apiRequest("/monitoring/camera", {
    method: "POST",
    body: formData,
  });
}

// ==========================================
// COMBINED AI ANALYSIS
// ==========================================

export async function analyzeSafety({
  audioFile,
  imageFile,
  hour,
  alone = false,
  darkArea = false,
}: {
  audioFile?: File;
  imageFile?: File;
  hour?: number;
  alone?: boolean;
  darkArea?: boolean;
}) {
  const formData = new FormData();

  if (audioFile) {
    formData.append("audio", audioFile);
  }

  if (imageFile) {
    formData.append("image", imageFile);
  }

  if (hour !== undefined) {
    formData.append("hour", String(hour));
  }

  formData.append(
    "alone",
    String(alone)
  );

  formData.append(
    "dark_area",
    String(darkArea)
  );

  return apiRequest("/monitoring/analyze", {
    method: "POST",
    body: formData,
  });
}