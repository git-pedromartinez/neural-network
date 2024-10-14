/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { Matrix } from "./matrix.models";
import { NeuralNetworkConfig } from "./neural-network-config.models";

export type InputLayer<T = number> = T[];

export type OutputLayer<T = number> = T[];

export interface NeuralNetworkData<T = number> {
  inputs: InputLayer<T>;
  targets: OutputLayer<T>;
}

export type TestingNetworkData = Omit<NeuralNetworkData, "targets">;

export type NetworkMetaData = Partial<
  NeuralNetworkConfig & {
    errorThreshold: number;
    trainingName: string;
  }
>;

export interface TrainingNetworkData {
  metaData: NetworkMetaData;
  weights: Matrix<"3D">;
  biases: Matrix<"2D">;
}
