import { createBrowserRouter } from "react-router-dom";
import CommandCenter from "@/pages/CommandCenter";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <CommandCenter />,
  },
]);
