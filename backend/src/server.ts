import express from "express";
import routes from "./routes.js";
import { startPodMonitor } from "./podMonitor.js";

const app = express();

app.use(express.json());

app.use(routes);

startPodMonitor();

app.listen(3000, () => {
  console.log("Server running on port 3000");
});
