import apiRequest from "./apiClient";

export interface HistoryItem {
  id: string;
  type: string;
  status: string;
  message: string;
  latitude: number | null;
  longitude: number | null;
  created_at: string | null;
  resolved_at: string | null;
}

export interface HistoryResponse {
  count: number;
  history: HistoryItem[];
}

export async function getHistory(): Promise<HistoryResponse> {
  return apiRequest("/history/", {
    method: "GET",
  });
}
