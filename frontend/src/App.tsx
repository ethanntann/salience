import { EvalDashboard } from "./components/EvalDashboard";
import { ReviewInbox } from "./components/ReviewInbox";

export function App() {
  if (window.location.pathname.startsWith("/eval")) {
    return <EvalDashboard />;
  }
  return <ReviewInbox />;
}
