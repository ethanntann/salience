import { afterEach, describe, expect, it, vi } from "vitest";

describe("resolveApiUrl", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
    vi.resetModules();
  });

  it("uses the backend origin on the documented Vite development port", async () => {
    const { resolveApiUrl } = await import("../src/api");

    expect(resolveApiUrl("/clips/1/video")).toBe("http://localhost:8000/clips/1/video");
  });

  it("uses VITE_API_BASE_URL and normalizes its trailing slash", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "https://api.example.test/");
    const { resolveApiUrl } = await import("../src/api");

    expect(resolveApiUrl("/clips/1/video")).toBe("https://api.example.test/clips/1/video");
  });

  it("keeps root-relative URLs same-origin when the API base is empty", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    const { resolveApiUrl } = await import("../src/api");

    expect(resolveApiUrl("/clips/1/video")).toBe("/clips/1/video");
  });
});
