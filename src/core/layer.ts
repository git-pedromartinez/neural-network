/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { Neuron } from "./neuron";
import {
  ActivationFunction,
  ActivationFunctionDerivative,
  Matrix,
} from "../models";

export class Layer {
  neurons: Neuron[];
  identifier: number;

  constructor(
    numNeurons: number,
    inputSize: number,
    activationFunction: ActivationFunction,
    activationDerivative: ActivationFunctionDerivative,
    identifier: number
  ) {
    this.identifier = identifier;
    this.neurons = Array.from(
      { length: numNeurons },
      (n, i) =>
        new Neuron(inputSize, activationFunction, activationDerivative, [identifier, i])
    );
  }

  forward(inputs: Matrix): Matrix {
    return this.neurons.map((neuron) => neuron.activate(inputs));
  }
}
