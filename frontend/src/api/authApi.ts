import apiRequest from "./apiClient";

// ==========================================
// REGISTER
// ==========================================

export interface RegisterData {
  name: string;
  phone: string;
  email: string;
  password: string;
}

// ==========================================
// LOGIN
// ==========================================

export interface LoginData {
  email: string;
  password: string;
}

// ==========================================
// USER RESPONSE
// ==========================================

export interface UserResponse {
  id?: string;
  name?: string;
  phone?: string;
  email?: string;
}

// ==========================================
// REGISTER RESPONSE
// ==========================================

export interface RegisterResponse {
  message?: string;

  user?: UserResponse;
}

// ==========================================
// LOGIN RESPONSE
// ==========================================

export interface LoginResponse {
  message?: string;

  access_token?: string;

  token_type?: string;

  user?: UserResponse;
}

// ==========================================
// REGISTER USER
// ==========================================

export async function registerUser(
  data: RegisterData
): Promise<RegisterResponse> {

  return apiRequest(
    "/auth/register",
    {
      method: "POST",

      body: JSON.stringify({
        name: data.name.trim(),

        phone: data.phone.trim(),

        email: data.email
          .trim()
          .toLowerCase(),

        password: data.password,
      }),
    }
  );
}

// ==========================================
// LOGIN USER
// ==========================================

export async function loginUser(
  data: LoginData
): Promise<LoginResponse> {

  const response =
    await apiRequest(
      "/auth/login",
      {
        method: "POST",

        body: JSON.stringify({
          email: data.email
            .trim()
            .toLowerCase(),

          password: data.password,
        }),
      }
    );

  // ----------------------------------------
  // Save JWT
  // ----------------------------------------

  if (response?.access_token) {

    localStorage.setItem(
      "access_token",
      response.access_token
    );

    // --------------------------------------
    // Save logged-in user
    // --------------------------------------

    localStorage.setItem(
      "user",
      JSON.stringify(
        response.user || {}
      )
    );
  }

  return response;
}

// ==========================================
// LOGOUT
// ==========================================

export function logoutUser() {

  localStorage.removeItem(
    "access_token"
  );

  localStorage.removeItem(
    "user"
  );
}

// ==========================================
// GET CURRENT USER
// ==========================================

export function getCurrentUser(): UserResponse | null {

  const user =
    localStorage.getItem("user");

  if (!user) {
    return null;
  }

  try {

    return JSON.parse(user);

  } catch {

    localStorage.removeItem(
      "user"
    );

    return null;
  }
}