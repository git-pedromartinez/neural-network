/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import {
  ActivationFunction,
  ActivationFunctionDerivative,
} from "./activation-functions.models";
import { Matrix } from "./matrix.models";

export interface NeuralNetworkConfig {
  sizes: Matrix;
  learningRate: number;
  epochs: number;
  activationFunction?: ActivationFunction;
  activationDerivative?: ActivationFunctionDerivative;
}
