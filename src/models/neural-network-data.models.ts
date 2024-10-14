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

/**
 * Experimental feature: simulator
 */
export type NetworkMetadata = Partial<
  NeuralNetworkConfig & {
    trainingName: string;
    errorThreshold: number;
  }
>;

export interface TrainingNetworkData {
  metaData?: NetworkMetadata;
  weights: Matrix<"3D">;
  biases: Matrix<"2D">;
}
