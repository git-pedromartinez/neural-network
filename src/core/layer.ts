/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { Neuron } from "./neuron";
import { ActivationFunction, ActivationFunctionDerivative } from "../models";

export class Layer {
  neurons: Neuron[];
  identifier: string;

  constructor(
    numNeurons: number,
    inputSize: number,
    activationFunction: ActivationFunction,
    activationDerivative: ActivationFunctionDerivative,
    identifier: string = "Layer"
  ) {
    this.identifier = identifier;
    this.neurons = Array.from(
      { length: numNeurons },
      (n, i) =>
        new Neuron(
          inputSize,
          activationFunction,
          activationDerivative,
          `${identifier},Neuron[${i}]`
        )
    );
  }

  forward(inputs: number[]): number[] {
    return this.neurons.map((neuron) => neuron.activate(inputs));
  }
}
