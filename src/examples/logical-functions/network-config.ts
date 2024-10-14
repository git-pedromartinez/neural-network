/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { NeuralNetworkConfig } from "../../models";
import { layers, learningRate, epochs } from "./constants";

export const network_config: NeuralNetworkConfig = {
  sizes: layers,
  learningRate: learningRate,
  epochs: epochs,
};
