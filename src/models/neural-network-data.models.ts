/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { Matrix } from "./matrix.models";

export type InputLayer<T = number> = T[];

export type OutputLayer<T = number> = T[];

export interface NeuralNetworkData<T = number> {
  inputs: InputLayer<T>;
  targets: OutputLayer<T>;
}

export type TestingNetworkData = Omit<NeuralNetworkData, "targets">;

export interface TrainingNetworkData {
  weights: Matrix<"3D">;
  biases: Matrix<"2D">;
}
