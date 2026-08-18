// ==========================================
// GuardAI - Monitoring API
// ==========================================

const API_BASE_URL =
  "http://127.0.0.1:8000";

// ==========================================
// COMMON REQUEST HELPER
// ==========================================

async function request(
  endpoint,
  options = {}
) {
  const token =
    localStorage.getItem(
      "access_token"
    );

  const headers = {
    Accept: "application/json",
    ...(options.headers || {}),
  };

  if (token) {
    headers.Authorization =
      `Bearer ${token}`;
  }

  const response = await fetch(
    `${API_BASE_URL}${endpoint}`,
    {
      ...options,
      headers,
    }
  );

  let data = null;

  try {
    data =
      await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        data?.message ||
        `Request failed with status ${response.status}`
    );
  }

  return data;
}

// ==========================================
// GET MONITORING STATUS
// ==========================================

export async function getMonitoringStatus() {
  return request(
    "/monitoring/status",
    {
      method: "GET",
    }
  );
}

// ==========================================
// START MONITORING
// ==========================================

export async function startMonitoring(
  settings = {}
) {
  return request(
    "/monitoring/start",
    {
      method: "POST",

      headers: {
        "Content-Type":
          "application/json",
      },

      body: JSON.stringify({
        voice_detection:
          settings.voice_detection ??
          true,

        emotion_detection:
          settings.emotion_detection ??
          true,

        camera_detection:
          settings.camera_detection ??
          true,
      }),
    }
  );
}

// ==========================================
// STOP MONITORING
// ==========================================

export async function stopMonitoring() {
  return request(
    "/monitoring/stop",
    {
      method: "POST",
    }
  );
}

// ==========================================
// CAMERA PERSON DETECTION
// ==========================================

export async function detectCamera(
  imageFile
) {
  if (!imageFile) {
    throw new Error(
      "Camera image is required."
    );
  }

  const formData =
    new FormData();

  formData.append(
    "image",
    imageFile
  );

  const token =
    localStorage.getItem(
      "access_token"
    );

  const headers = {
    Accept: "application/json",
  };

  if (token) {
    headers.Authorization =
      `Bearer ${token}`;
  }

  const response = await fetch(
    `${API_BASE_URL}/monitoring/camera`,
    {
      method: "POST",
      headers,
      body: formData,
    }
  );

  let data = null;

  try {
    data =
      await response.json();
  } catch {
    data = null;
  }

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        "Camera detection failed."
    );
  }

  return data;
}

// ==========================================
// CAMERA HISTORY
// ==========================================

export async function getCameraHistory() {
  return request(
    "/monitoring/camera/history",
    {
      method: "GET",
    }
  );
}

// ==========================================
// CAMERA IMAGE URL
// ==========================================

export function getCameraImageUrl(
  imageUrl
) {
  if (!imageUrl) {
    return "";
  }

  if (
    imageUrl.startsWith(
      "http://"
    ) ||
    imageUrl.startsWith(
      "https://"
    )
  ) {
    return imageUrl;
  }

  return `${API_BASE_URL}${
    imageUrl.startsWith("/")
      ? imageUrl
      : `/${imageUrl}`
  }`;
}

// ==========================================
// SAFETY / RISK ANALYSIS
// ==========================================

export async function analyzeSafety({
  // ----------------------------------------
  // Audio file
  // ----------------------------------------

  audioFile = null,

  // ----------------------------------------
  // Camera image
  // ----------------------------------------

  imageFile = null,

  // Backward compatibility:
  // Existing camera code may still send
  // "image" instead of "imageFile".
  image = null,

  // ----------------------------------------
  // Risk inputs
  // ----------------------------------------

  hour =
    new Date().getHours(),

  alone = false,

  darkArea = false,

  // ----------------------------------------
  // TEST MODE ONLY
  // ----------------------------------------

  distress_detected = false,

  fearful_emotion = false,
}) {
  const formData =
    new FormData();

  // ========================================
  // AUDIO
  // ========================================

  if (audioFile) {
    formData.append(
      "audio",
      audioFile
    );
  }

  // ========================================
  // CAMERA IMAGE
  // ========================================

  const finalImage =
    imageFile || image;

  if (finalImage) {
    formData.append(
      "image",
      finalImage
    );
  }

  // ========================================
  // BASIC RISK INPUTS
  // ========================================

  formData.append(
    "hour",
    String(hour)
  );

  formData.append(
    "alone",
    String(alone)
  );

  formData.append(
    "dark_area",
    String(darkArea)
  );

  // ========================================
  // TEST MODE
  // ========================================

  formData.append(
    "test_distress_detected",
    String(
      distress_detected
    )
  );

  formData.append(
    "test_fearful_emotion",
    String(
      fearful_emotion
    )
  );

  // ========================================
  // DEBUG
  // ========================================

  console.log(
    "GuardAI analyzeSafety request:",
    {
      hasAudio:
        !!audioFile,

      audioName:
        audioFile?.name ||
        null,

      audioType:
        audioFile?.type ||
        null,

      audioSize:
        audioFile?.size ||
        0,

      hasImage:
        !!finalImage,

      imageName:
        finalImage?.name ||
        null,

      hour,

      alone,

      darkArea,

      test_distress_detected:
        distress_detected,

      test_fearful_emotion:
        fearful_emotion,
    }
  );

  // ========================================
  // AUTHENTICATION
  // ========================================

  const token =
    localStorage.getItem(
      "access_token"
    );

  const headers = {
    Accept:
      "application/json",
  };

  if (token) {
    headers.Authorization =
      `Bearer ${token}`;
  }

  // IMPORTANT:
  // Do NOT manually set Content-Type here.
  // Browser automatically adds the correct
  // multipart/form-data boundary.

  // ========================================
  // API REQUEST
  // ========================================

  const response =
    await fetch(
      `${API_BASE_URL}/monitoring/analyze`,
      {
        method: "POST",

        headers,

        body: formData,
      }
    );

  // ========================================
  // PARSE RESPONSE
  // ========================================

  let data = null;

  try {
    data =
      await response.json();
  } catch {
    data = null;
  }

  // ========================================
  // ERROR HANDLING
  // ========================================

  if (!response.ok) {
    throw new Error(
      data?.detail ||
        data?.message ||
        "Safety analysis failed."
    );
  }

  // ========================================
  // DEBUG RESPONSE
  // ========================================

  console.log(
    "GuardAI analyzeSafety response:",
    data
  );

  return data;
}