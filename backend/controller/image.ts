import express from "express";
import { storeImage, processImage } from "../service/image";

const router = express.Router();

// 1️⃣ POST /uploadImage
router.post("/uploadImage", (req, res) => {
  const { userId, image } = req.body;

  if (!userId || !image) {
    return res.status(400).json({ error: "Missing userId or image" });
  }

  storeImage(userId, image);

  // Redirect user to SSE endpoint
  res.json({
    message: "Image uploaded",
    sse: `/image/imageReady?userId=${userId}`
  });
});


// 2️⃣ SSE /imageReady
router.get("/imageReady", async (req, res) => {
  const userId = req.query.userId as string;

  if (!userId) {
    return res.status(400).end();
  }

  // SSE headers
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  try {
    const resultImage = await processImage(userId);
    console.log(resultImage)
    // Send result
    res.write(`data: ${JSON.stringify({ image: resultImage })}\n\n`);

    res.end(); // close after sending once
  } catch (err: any) {
    res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
    res.end();
  }
});

export default router;