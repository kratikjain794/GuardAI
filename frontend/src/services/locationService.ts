import apiRequest from "../api/apiClient";

// ==========================================
// LOCATION
// ==========================================

export interface LocationData {
  latitude: number;
  longitude: number;
}

// ==========================================
// UPDATE CURRENT LOCATION
// ==========================================

export async function updateLocation(
  location: LocationData
) {
  return apiRequest("/location/", {
    method: "POST",
    body: JSON.stringify(location),
  });
}

// ==========================================
// GET LATEST LOCATION
// ==========================================

export async function getLatestLocation() {
  return apiRequest("/location/latest", {
    method: "GET",
  });
}