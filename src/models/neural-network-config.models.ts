/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import {
  ActivationFunction,
  ActivationFunctionDerivative,
} from "./activation-functions.models";

export interface NeuralNetworkConfig {
  sizes: number[];
  learningRate: number;
  epochs: number;
  activationFunction?: ActivationFunction;
  activationDerivative?: ActivationFunctionDerivative;
}
