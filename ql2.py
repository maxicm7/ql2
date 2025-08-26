import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Necesitas instalar deap si no lo tienes: pip install deap
from deap import base, creator, tools, algorithms

# ----------------------- Configuración Inicial DEAP -----------------------
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# ----------------------- Funciones Auxiliares -----------------------

@st.cache_data
def load_data_and_counts(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'Numero' in df.columns and 'Atraso' in df.columns:
                df['Numero'] = pd.to_numeric(df['Numero'], errors='coerce')
                df['Atraso'] = pd.to_numeric(df['Atraso'], errors='coerce')
                df.dropna(subset=['Numero', 'Atraso'], inplace=True)
                df['Numero'] = df['Numero'].astype(int).astype(str)
                df['Atraso'] = df['Atraso'].astype(int)

                if df.empty:
                     st.warning("No se encontraron filas válidas.")
                     return None, {}, {}, [], {}, 0

                st.success("Archivo de atrasos cargado exitosamente.")
                numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
                atrasos_disponibles_int = sorted(list(set(df['Atraso'].tolist())))
                numeros_validos = list(numero_a_atraso.keys())
                prob_por_numero = 1.0 / len(numeros_validos) if numeros_validos else 0
                distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}
                atraso_counts = df['Atraso'].astype(str).value_counts().to_dict()
                total_atraso_dataset = df['Atraso'].sum()

                return df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset
            else:
                 st.error("El archivo de atrasos debe contener 'Numero' y 'Atraso'.")
                 return None, {}, {}, [], {}, 0
        except Exception as e:
            st.error(f"Error al procesar el archivo de atrasos: {e}")
            return None, {}, {}, [], {}, 0
    return None, {}, {}, [], {}, 0

@st.cache_data
def load_historical_combinations(uploaded_file):
    if uploaded_file is not None:
        try:
            df_hist = pd.read_csv(uploaded_file, header=None)
            historical_sets = []
            for index, row in df_hist.iterrows():
                comb = pd.to_numeric(row, errors='coerce').dropna().astype(int).tolist()
                if len(comb) >= 6: # Asegurar que tenga al menos 6 números
                    historical_sets.append(set(comb))
            if not historical_sets:
                st.warning("El archivo de historial no contenía combinaciones válidas.")
                return []
            st.success(f"Archivo de historial cargado: {len(historical_sets)} combinaciones encontradas.")
            return historical_sets
        except Exception as e:
            st.error(f"Error al procesar el archivo de historial: {e}")
            return []
    return []

# --- NUEVO: Función para analizar el Cálculo Especial en el historial ---
@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    """Calcula la distribución del 'Cálculo Especial' para el set de datos histórico."""
    if not historical_sets or not numero_a_atraso or total_atraso_dataset is None:
        return None
    
    special_calc_values = []
    for comb_set in historical_sets:
        # Sumar solo si el número está en el mapa de atrasos
        suma_atrasos_comb = sum(numero_a_atraso.get(str(num), 0) for num in comb_set if str(num) in numero_a_atraso)
        valor_especial = total_atraso_dataset + 40 - suma_atrasos_comb
        special_calc_values.append(valor_especial)
        
    if not special_calc_values:
        return None
        
    return {
        "min": int(np.min(special_calc_values)),
        "max": int(np.max(special_calc_values)),
        "mean": int(np.mean(special_calc_values)),
        "std": int(np.std(special_calc_values))
    }

def generar_combinaciones_con_restricciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations, total_atraso_dataset, special_calc_range):
    valores = list(distribucion_probabilidad.keys())
    combinaciones = []
    intentos_totales = 0
    max_intentos = n_combinaciones * 200 # Aumentar margen por la nueva restricción

    while len(combinaciones) < n_combinaciones and intentos_totales < max_intentos:
        intentos_totales += 1
        seleccionados = []
        atrasos_seleccionados = Counter()
        usados = set()

        while len(seleccionados) < n_selecciones:
            valores_posibles = [
                v for v, p in distribucion_probabilidad.items() 
                if v not in usados and atrasos_seleccionados.get(str(numero_a_atraso.get(v)), 0) < restricciones_atraso.get(str(numero_a_atraso.get(v)), n_selecciones)
            ]
            if not valores_posibles: break
            nuevo_valor = random.choice(valores_posibles) # Simplificado a elección uniforme entre válidos
            seleccionados.append(nuevo_valor)
            usados.add(nuevo_valor)
            atraso = numero_a_atraso.get(nuevo_valor)
            if atraso is not None: atrasos_seleccionados[str(atraso)] += 1

        if len(seleccionados) == n_selecciones:
            es_valida_historial = True
            if historical_combinations:
                seleccionados_set = set(int(n) for n in seleccionados)
                for hist_comb_set in historical_combinations:
                    if len(seleccionados_set.intersection(hist_comb_set)) > 2:
                        es_valida_historial = False
                        break
            
            if es_valida_historial:
                # --- NUEVO: Validar si el Cálculo Especial está en el rango deseado ---
                suma_atrasos_comb = sum(numero_a_atraso.get(val, 0) for val in seleccionados)
                valor_especial = total_atraso_dataset + 40 - suma_atrasos_comb
                if special_calc_range[0] <= valor_especial <= special_calc_range[1]:
                    seleccionados.sort(key=int)
                    combinaciones.append(tuple(seleccionados))

    conteo_combinaciones = Counter(combinaciones)
    probabilidad_combinaciones = {}
    for combinacion, frecuencia in conteo_combinaciones.items():
        prob_comb = np.prod([distribucion_probabilidad.get(val, 0) for val in combinacion])
        probabilidad_combinaciones[combinacion] = (frecuencia, prob_comb)

    return sorted(probabilidad_combinaciones.items(), key=lambda x: (-x[1][1], -x[1][0]))


def procesar_combinaciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, n_ejecuciones, historical_combinations, total_atraso_dataset, special_calc_range):
    resultados = []
    task_args = (distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations, total_atraso_dataset, special_calc_range)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *task_args) for _ in range(n_ejecuciones)]
        for future in as_completed(futures):
            resultados.append(future.result())
    return resultados


def encontrar_combinaciones_coincidentes(resultados):
    if not resultados: return {}
    counts = Counter(comb for res in resultados for comb, _ in res)
    return {comb: [] for comb, count in counts.items() if count == len(resultados)}


def evaluar_individuo_deap(individuo, distribucion_prob, num_atraso, restr_atraso, n_sel, historical_combinations, total_atraso_dataset, special_calc_range):
    if len(individuo) != n_sel: return (0,)
    
    atrasos = Counter(num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None)
    for atraso, cantidad in atrasos.items():
        if cantidad > restr_atraso.get(str(atraso), n_sel): return (0,)

    if historical_combinations:
        individuo_set = set(int(n) for n in individuo)
        for hist_set in historical_combinations:
            if len(individuo_set.intersection(hist_set)) > 2: return (0,)
            
    # --- NUEVO: Penalizar si el Cálculo Especial está fuera de rango ---
    suma_atrasos = sum(num_atraso.get(val, 0) for val in individuo)
    valor_especial = total_atraso_dataset + 40 - suma_atrasos
    if not (special_calc_range[0] <= valor_especial <= special_calc_range[1]):
        return (0,)

    probabilidad = np.prod([distribucion_prob.get(val, 0) for val in individuo])
    return (probabilidad,)


def ejecutar_algoritmo_genetico(n_generaciones, n_poblacion, cxpb, mutpb, distribucion_prob, numero_a_atraso, restricciones_atraso, historical_combinations, total_atraso_dataset, special_calc_range, n_selecciones=6):
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initRepeat, creator.Individual, lambda: random.choice(list(distribucion_prob.keys())), n_selecciones)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # --- Pasar los nuevos parámetros a la función de evaluación ---
    toolbox.register("evaluate", evaluar_individuo_deap, distribucion_prob=distribucion_prob, num_atraso=numero_a_atraso, restr_atraso=restricciones_atraso, n_sel=n_selecciones, historical_combinations=historical_combinations, total_atraso_dataset=total_atraso_dataset, special_calc_range=special_calc_range)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=n_poblacion)
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, n_generaciones, verbose=False)
    
    best_ind = tools.selBest(population, k=1)[0] if population else None
    if best_ind:
        # Asegurarse de que no haya duplicados y tenga el tamaño correcto después de la evolución
        best_ind = sorted(list(set(best_ind)), key=int)
        if len(best_ind) != n_selecciones: return None, 0.0, "El AG no pudo mantener individuos válidos."
        best_fitness = toolbox.evaluate(best_ind)[0]
        return best_ind, best_fitness, None
    return None, 0.0, "La población final estaba vacía."

# ----------------------- Interfaz de Streamlit -----------------------

st.set_page_config(layout="wide")
st.title("Generador Avanzado de Combinaciones de Números")

st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Sube CSV de Atrasos ('Numero', 'Atraso')", type="csv", key="atraso_uploader")
    df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset = load_data_and_counts(uploaded_file)
with col2:
    historical_file = st.file_uploader("Sube CSV con Historial de Combinaciones (Opcional)", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(historical_file)

n_selecciones = 6
if df is not None:
     st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso_dataset}")

st.header("2. Configurar Parámetros y Restricciones")
restricciones_finales = {}
if atrasos_disponibles_int:
    with st.expander("Configurar Restricciones de Atraso"):
        # ... (código de restricciones de atraso sin cambios)
        selected_atrasos = st.multiselect("Selecciona 'Atraso' a restringir:", options=[str(a) for a in atrasos_disponibles_int], default=[str(a) for a in atrasos_disponibles_int])
        cols = st.columns(4)
        for i, atraso_str in enumerate(selected_atrasos):
            with cols[i % 4]:
                limit = st.number_input(f"Max Atraso '{atraso_str}':", min_value=0, max_value=n_selecciones, value=atraso_counts.get(atraso_str, 0), key=f"res_{atraso_str}")
                restricciones_finales[atraso_str] = limit

# --- NUEVO: Expander para el filtro del Cálculo Especial ---
special_calc_range = (0, 1000) # Un rango por defecto muy amplio
if historical_combinations_set and total_atraso_dataset is not None:
    with st.expander("Configurar Filtro de 'Cálculo Especial'", expanded=True):
        stats = analyze_historical_special_calc(historical_combinations_set, total_atraso_dataset, numero_a_atraso)
        if stats:
            st.info(f"Análisis del historial: El 'Cálculo Especial' varía de **{stats['min']}** a **{stats['max']}**, con un promedio de **{stats['mean']}**.")
            # Rango por defecto: promedio ± 1 desviación estándar
            default_range_start = stats['mean'] - stats['std']
            default_range_end = stats['mean'] + stats['std']
            
            special_calc_range = st.slider(
                "Selecciona el rango deseado para el 'Cálculo Especial':",
                min_value=stats['min'] - 50, # Dar un poco de margen
                max_value=stats['max'] + 50,
                value=(default_range_start, default_range_end)
            )
        else:
            st.warning("No se pudo analizar el historial para sugerir un rango.")

with st.expander("Configurar Parámetros de los Algoritmos"):
    # ... (código de parámetros de algoritmos sin cambios)
    col_ga, col_sim = st.columns(2)
    with col_ga: st.subheader("Algoritmo Genético"); ga_ngen=st.slider("Generaciones",10,1000,200); ga_npob=st.slider("Población",100,5000,1000); ga_cxpb=st.slider("Cruce",0.0,1.0,0.7); ga_mutpb=st.slider("Mutación",0.0,1.0,0.2)
    with col_sim: st.subheader("Simulación Concurrente"); sim_n_comb=st.number_input("Combinaciones/Ejec.",1000,value=100000); sim_n_ejec=st.number_input("Ejecuciones",1,value=8)

st.header("3. Ejecutar Algoritmos")
if not numero_a_atraso:
    st.warning("Carga un archivo de Atrasos para ejecutar.")
else:
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Ejecutar Algoritmo Genético"):
            with st.spinner("Buscando la mejor combinación..."):
                mejor_individuo, mejor_fitness, error_msg = ejecutar_algoritmo_genetico(
                    ga_ngen, ga_npob, ga_cxpb, ga_mutpb,
                    distribucion_probabilidad, numero_a_atraso, restricciones_finales,
                    historical_combinations_set, total_atraso_dataset, special_calc_range, n_selecciones
                )
            if error_msg: st.error(error_msg)
            elif mejor_individuo and mejor_fitness > 0:
                st.subheader("Mejor Combinación (GA)")
                st.success(f"**Combinación: {' - '.join(map(str, mejor_individuo))}**")
                suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in mejor_individuo)
                valor_especial = total_atraso_dataset + 40 - suma_atrasos
                st.write(f"**Cálculo Especial:** {valor_especial} (dentro del rango {special_calc_range})")
            else: st.warning("El GA no encontró una combinación válida que cumpliera todas las restricciones.")

    with run_col2:
        if st.button("Ejecutar Simulación Concurrente"):
            with st.spinner("Generando y procesando combinaciones en paralelo..."):
                resultados = procesar_combinaciones(
                    distribucion_probabilidad, numero_a_atraso, restricciones_finales,
                    n_selecciones, sim_n_comb, sim_n_ejec, historical_combinations_set,
                    total_atraso_dataset, special_calc_range
                )
                coincidentes = encontrar_combinaciones_coincidentes(resultados)
            
            st.subheader("Combinaciones Coincidentes (Simulación)")
            if coincidentes:
                data = []
                first_run_map = dict(resultados[0]) if resultados else {}
                for comb in coincidentes:
                    suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in comb)
                    valor_especial = total_atraso_dataset + 40 - suma_atrasos
                    data.append({
                        "Combinación": " - ".join(map(str, comb)),
                        "Probabilidad": first_run_map.get(comb, (0, 0.0))[1],
                        "Suma Atrasos": suma_atrasos,
                        "Cálculo Especial": valor_especial
                    })
                st.dataframe(pd.DataFrame(data).sort_values("Probabilidad", ascending=False))
            else:
                st.info("No se encontraron combinaciones que aparecieran en todas las simulaciones cumpliendo todas las restricciones.")

st.sidebar.header("Información")
st.sidebar.markdown("""
**...**
**NUEVO: Filtro de Cálculo Especial**
- Si cargas un archivo de historial, la aplicación lo analizará para encontrar el rango típico del "Cálculo Especial".
- Puedes usar el *slider* para definir un **rango objetivo**. Los algoritmos solo buscarán combinaciones que resulten en un "Cálculo Especial" dentro de este rango.
""")
