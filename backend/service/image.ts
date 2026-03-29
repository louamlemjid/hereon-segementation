import axios from "axios";
import { hashStore } from "../app";

export const storeImage = (userId: string, image: string) => {
  hashStore[userId] = image;
};

export const processImage = async (userId: string): Promise<string> => {
  const image = hashStore[userId];

  if (!image) {
    throw new Error("Image not found");
  }

  // 🔥 Call your external API
  const response = await axios.post("http://img-seg-container:5000/image/process", {
    image
  });

  return response.data.result; // assume { result: base64Image }
};