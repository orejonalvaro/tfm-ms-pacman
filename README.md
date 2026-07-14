# Ms. Pac-Man Reinforcement Learning Agents

Repositorio del TFM dedicado al entrenamiento, evaluación y comparación de agentes de aprendizaje por refuerzo profundo
en el entorno `ALE/MsPacman-v5`.

El objetivo del repositorio es conservar una version reproducible del trabajo experimental: código fuente,
configuraciones, modelos finales, resultados en CSV y figuras.

## Que Contiene

El repositorio permite trabajar con tres tipos de agentes:

- Agente aleatorio como linea base simple.
- DQN base entrenado con Stable-Baselines3.
- DQN con reward shaping para modificar la señal de recompensa durante el entrenamiento.

También incluye los modelos entrenados usados en la comparación final y los modelos del análisis de sensibilidad.

## Como Funciona

El flujo general del proyecto es:

1. Crear el entorno de Ms. Pac-Man con el preprocesado Atari definido en `make_env.py`.
2. Entrenar un agente DQN base con `train_dqn.py` o un agente con reward shaping con `train_dqn_reward.py`.
3. Guardar el modelo entrenado en `models/`.
4. Evaluar los modelos con `evaluate_model.py` usando siempre la recompensa original del entorno.
5. Guardar los resultados de evaluación en `results/`.
6. Analizar y comparar los resultados con `analyze_results.py`.
7. Guardar las figuras finales en `figures/`.

Durante el entrenamiento del agente con reward shaping se modifica la recompensa recibida por el modelo. Durante la
evaluación no se modifica la recompensa, para que todos los agentes sean comparables bajo el mismo criterio.

## Estructura Del Repositorio

```text
.
|-- README.md
|-- requirements.txt
|-- make_env.py
|-- train_dqn.py
|-- train_dqn_reward.py
|-- evaluate_random.py
|-- evaluate_model.py
|-- analyze_results.py
|-- config/
|-- wrappers/
|-- docs/
|-- models/
|   |-- 500k/
|   |-- 1m/
|-- results/
|   |-- 500k/
|   |-- 1m/
|   `-- sensitivity_500k/
`-- figures/
    |-- 500k/
    |-- 1m/
    `-- sensitivity_500k/
```

## Carpetas Principales

- `config/`: configuraciones YAML de los entrenamientos.
- `wrappers/`: wrapper de reward shaping.
- `models/`: modelos entrenados conservados para la memoria.
- `results/`: CSV de evaluación y comparación.
- `figures/`: gráficas generadas a partir de los resultados.
- `docs/`: notas resumidas del protocolo experimental y resultados.

## Instalación

Crear y activar un entorno virtual:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Entrenar Modelos

Entrenar DQN base:

```powershell
.\.venv\Scripts\python.exe .\train_dqn.py --total-timesteps 1000000 --seed 123 --model-path models\1m\dqn\dqn_1m_seed123.zip --checkpoint-dir logs\checkpoints\dqn_1m_seed123 --tensorboard-log logs\tensorboard_1m_dqn
```

Entrenar DQN con reward shaping:

```powershell
.\.venv\Scripts\python.exe .\train_dqn_reward.py --total-timesteps 1000000 --seed 123 --step-penalty-alpha 0.001 --life-loss-penalty-beta 1.0 --model-path models\1m\reward_alpha0p001_beta1\dqn_reward_1m_alpha0p001_beta1_seed123.zip --checkpoint-dir logs\checkpoints\reward_1m_seed123 --tensorboard-log logs\tensorboard_1m_reward
```

## Evaluar Modelos

Ejemplo de evaluación de un modelo entrenado:

```powershell
.\.venv\Scripts\python.exe .\evaluate_model.py --model-path models\1m\dqn\dqn_1m_seed123.zip --agent-name dqn_1m_seed123 --episodes 50 --seed 1000 --output-csv results\manual_eval\dqn_1m_seed123_eval.csv
```

El CSV generado contiene las recompensas y pasos por episodio. Estos CSV son la entrada para los análisis posteriores.

## Analizar Resultados

Los resultados ya consolidados están separados por experimento:

- `results/500k/`: comparaciones de modelos entrenados durante 500k timesteps.
- `results/1m/`: comparaciones de modelos entrenados durante 1M timesteps.
- `results/sensitivity_500k/`: análisis de sensibilidad de hiperparámetros de reward shaping.

Las figuras correspondientes están en:

- `figures/500k/`
- `figures/1m/`
- `figures/sensitivity_500k/`

## Convención De Nombres

```text
alpha0p001 = step_penalty_alpha 0.001
alpha0p01  = step_penalty_alpha 0.01
beta1      = life_loss_penalty_beta 1.0
beta0p5    = life_loss_penalty_beta 0.5
```

## Notas

- Los logs, checkpoints intermedios y TensorBoard no forman parte del repositorio.
- Los modelos incluidos son los necesarios para reproducir las evaluaciones usadas en la memoria.
