import express from "express";

const router = express.Router();


router.get("/", (req, res) => {
  

  // Redirect user to SSE endpoint
  res.json({
    message: "the test is working"
  });
});

export default router;