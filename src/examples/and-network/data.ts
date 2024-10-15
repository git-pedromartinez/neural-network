import { NeuralNetworkData } from "../../models";

// Training data for functions: XOR, AND, and OR
export const AND_DATA: NeuralNetworkData[] = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [0] },
  { inputs: [1, 0], targets: [0] },
  { inputs: [1, 1], targets: [1] },
];
