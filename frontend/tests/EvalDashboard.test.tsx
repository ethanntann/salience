import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { EvalClip, EvalClipResponse, EvalSummaryResponse, LabelEvalMetric } from "../src/types";
import { EvalDashboard } from "../src/components/EvalDashboard";

const { fetchEvalClips, fetchEvalSummary, saveTeacherLabelReview } = vi.hoisted(() => ({
  fetchEvalClips: vi.fn(),
  fetchEvalSummary: vi.fn(),
  saveTeacherLabelReview: vi.fn()
}));

vi.mock("../src/api", () => ({
  fetchEvalClips,
  fetchEvalSummary,
  resolveApiUrl: (path: string) => path,
  saveTeacherLabelReview
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

function clip(id: number, filename: string): EvalClip {
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
    feedback: [],
    thumbnail_variant: "one",
    teacher_provider: "test",
    teacher_confidence: 0.5,
    teacher_labels: { sniper_kill: "yes" },
    teacher_evidence: [],
    highlight_description: null,
    label_reviews: {},
    highlight_reviews: {},
    candidate_labels: { sniper_kill: "yes" },
    candidate_evidence: [],
    candidate_status: "pending",
    candidate_version: "test",
    candidate_created_at: null,
    candidate_event_audit: {
      available: false,
      highlight_description: null,
      primary_event: null,
      secondary_events: [],
      rejected_events: [],
      multi_kill: null,
      active_finish_count: null
    }
  };
}

function clipsPayload(mode: "candidate" | "live", items: EvalClip | EvalClip[]): EvalClipResponse {
  return { mode, labels: ["sniper_kill"], clips: Array.isArray(items) ? items : [items] };
}

function summary(reviewed: number): EvalSummaryResponse {
  const metric: LabelEvalMetric = {
    label_key: "sniper_kill",
    reviewed,
    teacher_yes: reviewed,
    expected_yes: reviewed,
    true_positive: reviewed,
    false_positive: 0,
    false_negative: 0,
    true_negative: 0,
    precision: 1,
    recall: 1,
    accuracy: 1
  };
  return { labels: [metric] };
}

describe("EvalDashboard request ordering", () => {
  beforeEach(() => {
    fetchEvalClips.mockReset();
    fetchEvalSummary.mockReset();
    saveTeacherLabelReview.mockReset();
  });

  it("keeps the newest load when responses arrive in reverse order", async () => {
    const staleClips = deferred<EvalClipResponse>();
    const staleSummary = deferred<EvalSummaryResponse>();
    const newestClips = deferred<EvalClipResponse>();
    const newestSummary = deferred<EvalSummaryResponse>();
    fetchEvalClips
      .mockResolvedValueOnce(clipsPayload("candidate", clip(1, "initial.mp4")))
      .mockReturnValueOnce(staleClips.promise)
      .mockReturnValueOnce(newestClips.promise);
    fetchEvalSummary
      .mockResolvedValueOnce(summary(1))
      .mockReturnValueOnce(staleSummary.promise)
      .mockReturnValueOnce(newestSummary.promise);

    render(<EvalDashboard />);
    await screen.findByText("initial.mp4");

    fireEvent.change(screen.getByLabelText("Review source"), { target: { value: "live" } });
    fireEvent.change(screen.getByLabelText("Review source"), { target: { value: "candidate" } });

    await act(async () => {
      newestClips.resolve(clipsPayload("candidate", clip(3, "newest.mp4")));
      newestSummary.resolve(summary(9));
      await Promise.all([newestClips.promise, newestSummary.promise]);
    });
    expect(await screen.findByText("newest.mp4")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();

    await act(async () => {
      staleClips.resolve(clipsPayload("live", clip(2, "stale.mp4")));
      staleSummary.resolve(summary(2));
      await Promise.all([staleClips.promise, staleSummary.promise]);
    });

    expect(screen.queryByText("stale.mp4")).not.toBeInTheDocument();
    expect(screen.getByText("newest.mp4")).toBeInTheDocument();
    expect(screen.getByText("9")).toBeInTheDocument();
  });

  it("serializes overlapping reviews so both cards commit in server order", async () => {
    const firstReview = deferred<EvalSummaryResponse>();
    const secondReview = deferred<EvalSummaryResponse>();
    saveTeacherLabelReview.mockReturnValueOnce(firstReview.promise).mockReturnValueOnce(secondReview.promise);
    fetchEvalClips.mockResolvedValueOnce(
      clipsPayload("candidate", [clip(1, "first.mp4"), clip(2, "second.mp4")])
    );
    fetchEvalSummary.mockResolvedValueOnce(summary(0));

    render(<EvalDashboard />);
    const firstCard = (await screen.findByText("first.mp4")).closest("article");
    const secondCard = screen.getByText("second.mp4").closest("article");
    expect(firstCard).not.toBeNull();
    expect(secondCard).not.toBeNull();

    fireEvent.click(within(firstCard!).getByRole("button", { name: "Expected yes" }));
    fireEvent.click(within(secondCard!).getByRole("button", { name: "Expected no" }));
    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(1));

    await act(async () => {
      firstReview.resolve(summary(1));
      await firstReview.promise;
    });
    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(2));

    await act(async () => {
      secondReview.resolve(summary(2));
      await secondReview.promise;
    });

    expect(within(firstCard!).getByRole("button", { name: "Expected yes" })).toHaveClass("selected");
    expect(within(secondCard!).getByRole("button", { name: "Expected no" })).toHaveClass("selected");
    expect(screen.getByText("reviewed total").previousElementSibling).toHaveTextContent("2");
  });

  it("continues the review queue after a rejection", async () => {
    const failedReview = deferred<EvalSummaryResponse>();
    saveTeacherLabelReview.mockReturnValueOnce(failedReview.promise).mockResolvedValueOnce(summary(1));
    fetchEvalClips.mockResolvedValueOnce(
      clipsPayload("candidate", [clip(1, "failed.mp4"), clip(2, "recovered.mp4")])
    );
    fetchEvalSummary.mockResolvedValueOnce(summary(0));

    render(<EvalDashboard />);
    const failedCard = (await screen.findByText("failed.mp4")).closest("article");
    const recoveredCard = screen.getByText("recovered.mp4").closest("article");

    fireEvent.click(within(failedCard!).getByRole("button", { name: "Expected yes" }));
    fireEvent.click(within(recoveredCard!).getByRole("button", { name: "Expected no" }));
    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(1));

    await act(async () => {
      failedReview.reject(new Error("review failed"));
      await failedReview.promise.catch(() => undefined);
    });

    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(2));
    expect(within(recoveredCard!).getByRole("button", { name: "Expected no" })).toHaveClass("selected");
    expect(screen.getByText("reviewed total").previousElementSibling).toHaveTextContent("1");
  });

  it("waits for a pending review before refreshing from the server", async () => {
    const reviewResponse = deferred<EvalSummaryResponse>();
    let reviewCommitted = false;
    saveTeacherLabelReview.mockImplementationOnce(async () => {
      const result = await reviewResponse.promise;
      reviewCommitted = true;
      return result;
    });
    fetchEvalClips
      .mockResolvedValueOnce(clipsPayload("candidate", clip(1, "initial.mp4")))
      .mockImplementationOnce(async () => {
        const refreshed = clip(1, reviewCommitted ? "after-review.mp4" : "stale.mp4");
        if (reviewCommitted) {
          refreshed.label_reviews.sniper_kill = "yes";
        }
        return clipsPayload("candidate", refreshed);
      });
    fetchEvalSummary
      .mockResolvedValueOnce(summary(0))
      .mockImplementationOnce(async () => summary(reviewCommitted ? 1 : 0));

    render(<EvalDashboard />);
    await screen.findByText("initial.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Expected yes" }));
    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Refresh sample" }));

    expect(fetchEvalClips).toHaveBeenCalledTimes(1);
    expect(fetchEvalSummary).toHaveBeenCalledTimes(1);

    await act(async () => {
      reviewResponse.resolve(summary(1));
      await reviewResponse.promise;
    });

    expect(await screen.findByText("after-review.mp4")).toBeInTheDocument();
    expect(screen.queryByText("stale.mp4")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Expected yes" })).toHaveClass("selected");
    expect(screen.getByText("reviewed total").previousElementSibling).toHaveTextContent("1");
  });

  it("prevents a review from starting until an in-flight refresh settles", async () => {
    const refreshedClips = deferred<EvalClipResponse>();
    const refreshedSummary = deferred<EvalSummaryResponse>();
    fetchEvalClips
      .mockResolvedValueOnce(clipsPayload("candidate", clip(1, "initial.mp4")))
      .mockReturnValueOnce(refreshedClips.promise);
    fetchEvalSummary
      .mockResolvedValueOnce(summary(0))
      .mockReturnValueOnce(refreshedSummary.promise);
    saveTeacherLabelReview.mockResolvedValueOnce(summary(1));

    render(<EvalDashboard />);
    await screen.findByText("initial.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Refresh sample" }));
    const reviewButton = screen.getByRole("button", { name: "Expected yes" });
    expect(reviewButton).toBeDisabled();

    fireEvent.click(reviewButton);
    expect(saveTeacherLabelReview).not.toHaveBeenCalled();

    await act(async () => {
      refreshedClips.resolve(clipsPayload("candidate", clip(1, "refreshed.mp4")));
      refreshedSummary.resolve(summary(0));
      await Promise.all([refreshedClips.promise, refreshedSummary.promise]);
    });

    const enabledReviewButton = screen.getByRole("button", { name: "Expected yes" });
    expect(await screen.findByText("refreshed.mp4")).toBeInTheDocument();
    expect(enabledReviewButton).toBeEnabled();
    fireEvent.click(enabledReviewButton);

    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(1));
    expect(enabledReviewButton).toHaveClass("selected");
  });

  it("ignores a review response after the evaluation mode changes", async () => {
    const reviewResponse = deferred<EvalSummaryResponse>();
    saveTeacherLabelReview.mockReturnValueOnce(reviewResponse.promise);
    fetchEvalClips
      .mockResolvedValueOnce(clipsPayload("candidate", clip(1, "candidate.mp4")))
      .mockResolvedValueOnce(clipsPayload("live", clip(2, "live.mp4")));
    fetchEvalSummary.mockResolvedValueOnce(summary(1)).mockResolvedValueOnce(summary(7));

    render(<EvalDashboard />);
    await screen.findByText("candidate.mp4");

    fireEvent.click(screen.getByRole("button", { name: "Expected yes" }));
    await waitFor(() => expect(saveTeacherLabelReview).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText("Review source"), { target: { value: "live" } });
    expect(fetchEvalClips).toHaveBeenCalledTimes(1);

    await act(async () => {
      reviewResponse.resolve(summary(99));
      await reviewResponse.promise;
    });

    expect(await screen.findByText("live.mp4")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.queryByText("99")).not.toBeInTheDocument();
    expect(screen.getByText("live.mp4")).toBeInTheDocument();
  });

  it("shows structured audit unavailable for legacy candidates", async () => {
    fetchEvalClips.mockResolvedValueOnce(clipsPayload("candidate", clip(1, "legacy.mp4")));
    fetchEvalSummary.mockResolvedValueOnce(summary(0));

    render(<EvalDashboard />);

    expect(await screen.findByText("Structured audit unavailable")).toBeInTheDocument();
    expect(screen.getByText("Raw candidate evidence")).toBeInTheDocument();
  });
});
