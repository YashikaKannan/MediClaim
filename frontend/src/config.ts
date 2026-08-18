// API configuration for the MediClaim platform
export const getApiUrl = (): string => {
  const savedUrl = localStorage.getItem('mediclaim_api_url');
  return savedUrl || 'http://localhost:8000';
};

export const setApiUrl = (url: string): void => {
  localStorage.setItem('mediclaim_api_url', url);
};
