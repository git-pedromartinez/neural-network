/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { ActivationFunctions } from "./activation-functions";
import { ActivationFunction, ActivationFunctionDerivative } from "../models";

export class Neuron {
  identifier: string;
  weights: number[];
  bias: number;
  activationFunction: ActivationFunction;
  activationDerivative: ActivationFunctionDerivative;

  constructor(
    inputSize: number,
    activationFunction: ActivationFunction = ActivationFunctions.sigmoid,
    activationDerivative: ActivationFunctionDerivative = ActivationFunctions.sigmoidDerivative,
    identifier: string = "Neuron"
  ) {
    this.identifier = identifier;
    this.weights = Array.from({ length: inputSize }, () => Math.random() - 0.5);
    this.bias = Math.random() - 0.5;
    this.activationFunction = activationFunction;
    this.activationDerivative = activationDerivative;
  }

  activate(inputs: number[]): number {
    const weightedSum: number = inputs.reduce(
      (sum, input, index) => sum + input * this.weights[index],
      this.bias
    );
    return this.activationFunction(weightedSum);
  }
}
