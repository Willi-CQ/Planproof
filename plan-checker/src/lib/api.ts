/** Public API origin only. Never place Gemini or NVIDIA provider keys in VITE_* variables. */
export const stoaApiUrl = (import.meta.env.VITE_STOA_API_BASE_URL ?? '').trim().replace(/\/+$/, '');
