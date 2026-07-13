import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Clip, TrainingStatus } from "../src/types";
import { ReviewInbox } from "../src/components/ReviewInbox";

const { fetchClips, fetchTeacherRunStatus, fetchTrainingStatus, scanFolder, sendFeedback, startTeacherRun } = vi.hoisted(() => ({
  fetchClips: vi.fn(),
  fetchTeacherRunStatus: vi.fn(),
  fetchTrainingStatus: vi.fn(),
  scanFolder: vi.fn(),
  sendFeedback: vi.fn(),
  startTeacherRun: vi.fn()
}));

vi.mock("../src/api", () => ({
  exportClips: vi.fn(),
  fetchClips,
  fetchTeacherRunStatus,
  fetchTrainingStatus,
  importLikedFolder: vi.fn(),
  resetTasteProfile: vi.fn(),
  resolveApiUrl: (path: string) => path,
  saveTasteProfile: vi.fn(),
  scanFolder,
  sendFeedback,
  startTeacherRun
}));

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((fulfill, fail) => {
    resolve = fulfill;
    reject = fail;
  });
  return { promise, reject, resolve };
}

function clip(id: number, filename: string, feedback: string[] = []): Clip {
  return {
    id,
    filename,
    path: filename,
    duration_sec: null,
    width: null,
    height: null,
    fps: null,
    size_bytes: null,
    source: "test",
    video_url: `/clips/${id}/video`,
    final_score: 0.5,
    base_score: 0.5,
    personal_score: null,
    confidence: 0.5,
    explanation: null,
    tags: [],
    feedback,
    thumbnail_variant: "one",
    teacher_provider: "fireworks",
    teacher_confidence: null,
    teacher_labels: {},
    teacher_evidence: [],
    highlight_description: null
  };
}

function clipInSet(id: number, filename: string, path: string, provider: string, source = "local"): Clip {
  return { ...clip(id, filename), path, teacher_provider: provider, source };
}

function training(positiveCount: number): TrainingStatus {
  return {
    clips: 2,
    teacher_labeled: 0,
    teacher_pending: 2,
    teacher_progress: 0,
    feedback_count: positiveCount,
    positive_count: positiveCount,
    negative_count: 0,
    positive_avg_personal_score: null,
    negative_avg_personal_score: null,
    personal_score_separation: null
  };
}

describe("ReviewInbox feedback ordering", () => {
  beforeEach(() => {
    fetchClips.mockReset();
    fetchTrainingStatus.mockReset();
    fetchTeacherRunStatus.mockReset();
    scanFolder.mockReset();
    sendFeedback.mockReset();
    startTeacherRun.mockReset();
  });

  it("serializes feedback and training refreshes in FIFO order", async () => {
    const firstResponse = deferred<Clip[]>();
    const firstTraining = deferred<TrainingStatus>();
    const secondResponse = deferred<Clip[]>();
    const secondTraining = deferred<TrainingStatus>();
    fetchClips.mockResolvedValueOnce([clip(1, "one.mp4"), clip(2, "two.mp4")]);
    fetchTrainingStatus
      .mockResolvedValueOnce(training(0))
      .mockReturnValueOnce(firstTraining.promise)
      .mockReturnValueOnce(secondTraining.promise);
    sendFeedback.mockReturnValueOnce(firstResponse.promise).mockReturnValueOnce(secondResponse.promise);

    render(<ReviewInbox />);
    const firstCard = (await screen.findByText("one.mp4")).closest("article");
    const secondCard = screen.getByText("two.mp4").closest("article");
    expect(firstCard).not.toBeNull();
    expect(secondCard).not.toBeNull();

    fireEvent.click(within(firstCard!).getByRole("button", { name: "Favorite" }));
    fireEvent.click(within(secondCard!).getByRole("button", { name: "Keep" }));
    await waitFor(() => expect(sendFeedback).toHaveBeenCalledTimes(1));

    await act(async () => {
      firstResponse.resolve([clip(1, "favorite-one.mp4", ["favorite"]), clip(2, "favorite-two.mp4")]);
      await firstResponse.promise;
    });
    expect(await screen.findByText("favorite-one.mp4")).toBeInTheDocument();
    expect(fetchTrainingStatus).toHaveBeenCalledTimes(2);
    expect(sendFeedback).toHaveBeenCalledTimes(1);

    await act(async () => {
      firstTraining.resolve(training(1));
      await firstTraining.promise;
    });
    await waitFor(() => expect(sendFeedback).toHaveBeenCalledTimes(2));

    await act(async () => {
      secondResponse.resolve([
        clip(1, "both-one.mp4", ["favorite"]),
        clip(2, "both-two.mp4", ["keep"])
      ]);
      await secondResponse.promise;
    });
    expect(await screen.findByText("both-one.mp4")).toBeInTheDocument();

    await act(async () => {
      secondTraining.resolve(training(2));
      await secondTraining.promise;
    });

    expect(screen.getByText("both-one.mp4")).toBeInTheDocument();
    expect(screen.getByText("favorite")).toBeInTheDocument();
    expect(screen.getByText("keep")).toBeInTheDocument();
    expect(screen.getByText("2 keep signals")).toBeInTheDocument();
    expect(fetchTrainingStatus).toHaveBeenCalledTimes(3);
  });

  it("continues the feedback queue after an operation rejects", async () => {
    const failedResponse = deferred<Clip[]>();
    fetchClips.mockResolvedValueOnce([clip(1, "one.mp4"), clip(2, "two.mp4")]);
    fetchTrainingStatus.mockResolvedValueOnce(training(0)).mockResolvedValueOnce(training(1));
    sendFeedback
      .mockReturnValueOnce(failedResponse.promise)
      .mockResolvedValueOnce([clip(1, "recovered-one.mp4"), clip(2, "recovered-two.mp4")]);

    render(<ReviewInbox />);
    const firstCard = (await screen.findByText("one.mp4")).closest("article");
    const secondCard = screen.getByText("two.mp4").closest("article");

    fireEvent.click(within(firstCard!).getByRole("button", { name: "Favorite" }));
    await waitFor(() => expect(sendFeedback).toHaveBeenCalledTimes(1));

    await act(async () => {
      failedResponse.reject(new Error("feedback failed"));
      await failedResponse.promise.catch(() => undefined);
    });

    await waitFor(() => expect(within(firstCard!).getByRole("button", { name: "Favorite" })).toBeEnabled());
    expect(screen.getByText("feedback failed")).toBeInTheDocument();

    fireEvent.click(within(secondCard!).getByRole("button", { name: "Keep" }));
    await waitFor(() => expect(sendFeedback).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("recovered-one.mp4")).toBeInTheDocument();
    expect(screen.getByText("1 keep signals")).toBeInTheDocument();
  });

  it("loads precomputed local-student samples without starting hosted inference", async () => {
    fetchClips
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        clipInSet(1, "new-clip.mp4", "/app/demo-video/new-clip.mp4", "local")
      ]);
    fetchTrainingStatus.mockResolvedValueOnce(training(0)).mockResolvedValueOnce(training(0));
    scanFolder.mockResolvedValueOnce({
      indexed: 1,
      total_found: 1,
      clips: [clipInSet(1, "new-clip.mp4", "/app/demo-video/new-clip.mp4", "local")]
    });
    render(<ReviewInbox />);
    await screen.findByText("Process new clips");

    fireEvent.change(screen.getByLabelText("Clip source"), {
      target: { value: "/app/demo-video" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Process clips" }));

    await waitFor(() => expect(scanFolder).toHaveBeenCalledWith("/app/demo-video", false));
    expect(startTeacherRun).not.toHaveBeenCalled();
    expect(await screen.findByText("new-clip.mp4")).toBeInTheDocument();
    expect(
      screen.getByText("Loaded 1 preprocessed local-student result(s); no cloud inference needed.")
    ).toBeInTheDocument();
  });

  it("filters the inbox between teacher, sample, and demo clip sets", async () => {
    fetchClips.mockResolvedValueOnce([
      clipInSet(1, "teacher.mp4", "/clips/teacher.mp4", "fireworks"),
      clipInSet(2, "sample.mp4", "/app/sample-clips/sample.mp4", "local"),
      clipInSet(3, "demo.mp4", "/app/demo-video/demo.mp4", "local")
    ]);
    fetchTrainingStatus.mockResolvedValueOnce(training(0));

    render(<ReviewInbox />);
    expect(await screen.findByText("teacher.mp4")).toBeInTheDocument();
    expect(screen.queryByText("sample.mp4")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Sample clips 1/ }));
    expect(screen.getByText("sample.mp4")).toBeInTheDocument();
    expect(screen.queryByText("teacher.mp4")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Demo clips 1/ }));
    expect(screen.getByText("demo.mp4")).toBeInTheDocument();
    expect(screen.queryByText("sample.mp4")).not.toBeInTheDocument();
  });

  it("opens the seeded demo set when no teacher-ranked clips exist", async () => {
    fetchClips.mockResolvedValueOnce([
      clipInSet(1, "seeded-demo.mp4", "demo://seeded-demo.mp4", "seeded-fireworks-teacher", "demo")
    ]);
    fetchTrainingStatus.mockResolvedValueOnce(training(0));

    render(<ReviewInbox />);

    expect(await screen.findByText("seeded-demo.mp4")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Demo clips 1/ })).toHaveAttribute("aria-pressed", "true");
  });
});
