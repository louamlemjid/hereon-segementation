import express from "express";
import imageRouter from "./controller/image";
import testRouter from "./controller/test";
import rootRouter from "./controller/root";
import cors from 'cors'

const app = express();
app.use(express.json({ limit: "10mb" }));

app.use(cors()); // allow all origins
// ✅ Global store
export const hashStore: Record<string, string> = {}; 
// userId -> base64 image

// Routes
app.use("/image", imageRouter);
app.use("/test",testRouter);
app.use("/",rootRouter);

const PORT = 8000;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});