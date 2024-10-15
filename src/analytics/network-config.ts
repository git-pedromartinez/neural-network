export const network_config = {
  sizes: [2, 300, 1],
  learningRate: 0.15,
  epochs: 10000,
};

export const XOR_DATA = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] },
];

export const XOR_TRAINING = "XOR_TRAINING";
