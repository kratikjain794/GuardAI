import apiRequest from "./apiClient";

// ==========================================
// Emergency Contact
// ==========================================

export interface EmergencyContact {
  name: string;
  phone: string;
  relation?: string | null;
}

export interface EmergencyContactResponse {
  id?: string;
  _id?: string;
  name: string;
  phone: string;
  relation?: string | null;
}

// ==========================================
// SOS
// ==========================================

export interface SOSRequest {
  latitude: number;
  longitude: number;
  message?: string;
}

export interface SOSLocation {
  latitude: number;
  longitude: number;
}

export interface SOSContactAlert {
  contact_id: string;
  name: string;
  phone: string;
  relation?: string | null;
  alert_status: string;
  message: string;
}

export interface SOSResponse {
  message?: string;
  sos_id?: string;
  status?: string;

  location?: SOSLocation;

  emergency_contacts?: {
    count: number;
    contacts: SOSContactAlert[];
  };

  notification?: {
    status: string;
    message: string;
    delivery: string;
  };
}

// ==========================================
// ADD EMERGENCY CONTACT
// ==========================================

export async function addEmergencyContact(
  contact: EmergencyContact
) {
  return apiRequest(
    "/contacts/",
    {
      method: "POST",
      body: JSON.stringify({
        name: contact.name,
        phone: contact.phone,
        relation:
          contact.relation || null,
      }),
    }
  );
}

// ==========================================
// GET ALL EMERGENCY CONTACTS
// ==========================================

export async function getEmergencyContacts() {
  return apiRequest(
    "/contacts/"
  );
}

// ==========================================
// GET SINGLE EMERGENCY CONTACT
// ==========================================

export async function getEmergencyContact(
  contactId: string
) {
  return apiRequest(
    `/contacts/${contactId}`
  );
}

// ==========================================
// DELETE EMERGENCY CONTACT
// ==========================================

export async function deleteEmergencyContact(
  contactId: string
) {
  return apiRequest(
    `/contacts/${contactId}`,
    {
      method: "DELETE",
    }
  );
}

// ==========================================
// TRIGGER SOS
// ==========================================

export async function triggerSOS(
  data: SOSRequest
): Promise<SOSResponse> {

  if (
    typeof data.latitude !==
      "number" ||
    typeof data.longitude !==
      "number"
  ) {
    throw new Error(
      "Valid location is required to activate SOS."
    );
  }

  return apiRequest(
    "/sos/",
    {
      method: "POST",

      body: JSON.stringify({
        latitude:
          data.latitude,

        longitude:
          data.longitude,

        ...(data.message
          ? {
              message:
                data.message,
            }
          : {}),
      }),
    }
  );
}

// ==========================================
// GET SOS
// ==========================================

export async function getSOS(
  sosId: string
): Promise<SOSResponse> {
  return apiRequest(
    `/sos/${sosId}`
  );
}

// ==========================================
// RESOLVE SOS
// ==========================================

export async function resolveSOS(
  sosId: string
) {
  return apiRequest(
    `/sos/${sosId}/resolve`,
    {
      method: "PATCH",
    }
  );
}