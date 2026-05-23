import { createBrowserRouter } from "react-router-dom";
import CommandCenter from "@/pages/CommandCenter";
import AgentConsole from "@/pages/AgentConsole";
import ResearchLab from "@/pages/ResearchLab";

export const router = createBrowserRouter([
  { path: "/", element: <CommandCenter /> },
  { path: "/agent", element: <AgentConsole /> },
  { path: "/research", element: <ResearchLab /> },
]);
