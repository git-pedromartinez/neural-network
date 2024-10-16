/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

import { ActivationFunctions } from "./activation-functions";
import {
  ActivationFunction,
  ActivationFunctionDerivative,
  Matrix,
} from "../models";

export class Neuron {
  identifier: Matrix;
  weights: Matrix;
  bias: number;
  activationFunction: ActivationFunction;
  activationDerivative: ActivationFunctionDerivative;

  constructor(
    inputSize: number,
    activationFunction: ActivationFunction = ActivationFunctions.sigmoid,
    activationDerivative: ActivationFunctionDerivative = ActivationFunctions.sigmoidDerivative,
    identifier: Matrix
  ) {
    this.identifier = identifier;
    this.weights = Array.from({ length: inputSize }, () => Math.random() - 0.5);
    this.bias = Math.random() - 0.5;
    this.activationFunction = activationFunction;
    this.activationDerivative = activationDerivative;
  }

  activate(inputs: Matrix): number {
    const weightedSum: number = inputs.reduce(
      (sum, input, index) => sum + input * this.weights[index],
      this.bias
    );
    return this.activationFunction(weightedSum);
  }
}
