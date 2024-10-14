import { NetworkMetaData } from "./neural-network-data.models";

/**
 * Experimental feature: simulator
 */
export type InputData = {
  input: number[];
  data: {
    targents: number[];
    outputs: number[];
    errors: number[][];
    weights: number[][][];
    biases: number[][];
    gradients: number[][][];
  };
}[];

export interface TrainingHistory {
  metaData: NetworkMetaData;
  epochs: Array<InputData>;
}
