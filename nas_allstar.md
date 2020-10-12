Existing NAS solutions supported by Retiarii
| NAS Solution                                  | Model Space                                                                   | Exploration Strategy                      | Input Mutator | Operator Mutator | Inserting Mutator | Customized Mutator | Implemented in Artifact |
|-----------------------------------------------|-------------------------------------------------------------------------------|-------------------------------------------|---------------|------------------|-------------------|--------------------|-------------------------|
| MnasNet                                       | MobileNetV2-based space                                                       | Reinforcement Learning                    |               | √                | √                 |                    | √                       |
| ENAS-CNN                                      | NASNet cell variant                                                           | Reinforcement Learning                    | √             | √                |                   |                    |                         |
| NASNet                                        | NASNet cell                                                                   | Reinforcement Learning                    | √             | √                |                   |                    | √                       |
| AmoebaNet                                     | NASNet cell                                                                   | Evolutionary                              | √             | √                |                   |                    | √                       |
| Single-Path One Shot (SPOS)                   | ShuffleNetV2-based space                                                      | Evolutionary                              |               | √                |                   |                    | √                       |
| Weight Agnostic Networks                      | Evolving space w/ inserting node, adding connection and altering   activation | and altering activation                   |               | √                |                   | √                  | √                       |
| Path-level   NAS                              | Evolving space w/ replication and split                                       | Reinforcement Learning                    |               |                  |                   | √                  | √                       |
| Large-Scale   Evolution                       | Evolving space w/   adding/removing/altering nodes adding/removing edges      | Evolutionary                              |               | √                | √                 | √                  |                         |
| EfficientNet                                  | MobileNetV2-based space                                                       | Reinforcement Learning                    |               | √                | √                 |                    |                         |
| Progressive NAS                               | NASNet cell variant                                                           | Evolutionary                              | √             | √                | √                 |                    | √                       |
| ENAS-RNN                                      | ENAS-RNN cell                                                                 | Reinforcement Learning                    | √             | √                |                   |                    |                         |
| Genetic CNN                                   | Genetic CNN space                                                             | Evolutionary                              | √             |                  |                   |                    |                         |
| Simplified One-Shot                           | Simplified one-shot space                                                     | Random Search                             | √             | √                |                   |                    |                         |
| DARTS, DARTS+, GDAS,P-DARTS, PC-DARTS,   SNAS | NASNet cell variant                                                           | Differentiable                            | √             | √                |                   |                    | √ (DARTS)               |
| DARTS-Language                                | ENAS-RNN cell                                                                 | Differentiable                            | √             | √                |                   |                    |                         |
| FBNet                                         | FBNet space                                                                   | Differentiable                            |               | √                |                   |                    | √                       |
| ProxylessNAS                                  | ProxylessNAS space                                                            | Differentiable / Reinforcement   Learning |               | √                |                   |                    | √                       |
| FairNAS                                       | ProxylessNAS space                                                            | Evolutionary                              |               | √                |                   |                    |                         |
| SCARLET-NAS                                   | ProxylessNAS space                                                            | Evolutionary                              |               | √                |                   |                    |                         |
| Single-Path NAS                               | MobileNetV2-based space                                                       | Differentiable                            |               | √                | √                 |                    | √                       |
| NAS-Bench-101                                 | NAS-Bench-101 space                                                           | N.A.                                      | √             | √                |                   |                    | √                       |
| NAS-Bench-201                                 | NAS-Bench-201 space                                                           | N.A.                                      |               | √                | √                 |                    | √                       |
| Once-For-All                                  | ProxylessNAS space                                                            | Performance predictor / Evolutionary      |               | √                |                   |                    | √                       |
| Hierarchical Representation                   | Hierarchical space                                                            | Evolutionary                              |               | √                | √                 |                    | √                       |
| TextNAS                                       | TextNAS space                                                                 | Reinforcement Learning                    | √             | √                |                   |                    | √                       |
| ChamNet                                       | MobileNetV2-based space                                                       | Performance predictor                     |               | √                | √                 |                    | √                       |
| Neural Predictor                              | NAS-Bench-101 space / ProxylessNAS space                                      | Performance predictor                     | √             | √                |                   |                    |                         |