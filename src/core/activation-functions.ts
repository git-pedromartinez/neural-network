/**
 * Author: Pedro Martinez
 * Email: id.pedromartinez@gmail.com
 * Position: Senior Software Engineer
 */

export class ActivationFunctions {
  static sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  static sigmoidDerivative(output: number): number {
    return output * (1 - output);
  }

  static relu(x: number): number {
    return Math.max(0, x);
  }

  static reluDerivative(output: number): number {
    return output > 0 ? 1 : 0;
  }

  static tanh(x: number): number {
    return Math.tanh(x);
  }

  static tanhDerivative(output: number): number {
    return 1 - output * output;
  }

  static binaryStep(x: number, limit = 0.5): number {
    return x >= limit ? 1 : 0;
  }
}
