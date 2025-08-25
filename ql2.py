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
# Estos deben crearse una sola vez. Streamlit ejecuta el script completo en cada interacción,
# por lo que verificamos si ya existen para evitar errores.
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
                     st.warning("No se encontraron filas válidas con 'Numero' y 'Atraso' numérico.")
                     return None, {}, {}, [], {}, 0

                st.success("Archivo de atrasos cargado exitosamente.")
                st.dataframe(df.head())

                numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
                atrasos_disponibles_int = sorted(list(set(df['Atraso'].tolist())))
                numeros_validos = list(numero_a_atraso.keys())
                prob_por_numero = 1.0 / len(numeros_validos) if numeros_validos else 0
                distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}
                atraso_counts = df['Atraso'].astype(str).value_counts().to_dict()
                total_atraso_dataset = df['Atraso'].sum()

                return df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset
            else:
                 st.error("El archivo de atrasos debe contener las columnas 'Numero' y 'Atraso'.")
                 return None, {}, {}, [], {}, 0
        except Exception as e:
            st.error(f"Error al procesar el archivo de atrasos: {e}")
            return None, {}, {}, [], {}, 0
    return None, {}, {}, [], {}, 0

# --- NUEVA FUNCIÓN PARA CARGAR EL HISTORIAL DE COMBINACIONES ---
@st.cache_data
def load_historical_combinations(uploaded_file):
    """Carga un CSV con combinaciones históricas y las devuelve como una lista de sets."""
    if uploaded_file is not None:
        try:
            # Leemos el CSV sin encabezado, cada columna es un número
            df_hist = pd.read_csv(uploaded_file, header=None)
            historical_sets = []
            for index, row in df_hist.iterrows():
                # Convertir cada número de la fila a numérico, ignorando errores (ej. celdas vacías)
                comb = pd.to_numeric(row, errors='coerce').dropna().astype(int).tolist()
                if comb: # Solo añadir si la fila no está vacía
                    historical_sets.append(set(comb))

            if not historical_sets:
                st.warning("El archivo de historial fue cargado pero no se encontraron combinaciones válidas.")
                return []

            st.success(f"Archivo de historial cargado: Se encontraron {len(historical_sets)} combinaciones anteriores.")
            # Muestra una pequeña muestra para confirmación
            st.write("Muestra de combinaciones cargadas del historial:")
            st.dataframe([list(s) for s in historical_sets[:5]])
            return historical_sets
        except Exception as e:
            st.error(f"Error al procesar el archivo de historial: {e}")
            return []
    return []

def generar_combinaciones_con_restricciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations):
    """
    Genera combinaciones, ahora también aplicando la restricción del historial (regla 2).
    """
    valores = list(distribucion_probabilidad.keys())
    combinaciones = []
    intentos_totales = 0
    max_intentos = n_combinaciones * 100  # Límite para evitar bucles infinitos

    while len(combinaciones) < n_combinaciones and intentos_totales < max_intentos:
        intentos_totales += 1
        seleccionados = []
        atrasos_seleccionados = Counter()
        usados = set()

        while len(seleccionados) < n_selecciones:
            valores_posibles = []
            probabilidades_posibles = []
            for valor, prob in distribucion_probabilidad.items():
                 if valor not in usados:
                     atraso = numero_a_atraso.get(valor)
                     if atraso is not None and atrasos_seleccionados.get(str(atraso), 0) < restricciones_atraso.get(str(atraso), n_selecciones):
                          valores_posibles.append(valor)
                          probabilidades_posibles.append(prob)

            if not valores_posibles: break
            total_prob = sum(probabilidades_posibles)
            if total_prob == 0: break
            probabilidades_posibles_normalized = [p / total_prob for p in probabilidades_posibles]
            nuevo_valor = random.choices(valores_posibles, weights=probabilidades_posibles_normalized, k=1)[0]

            seleccionados.append(nuevo_valor)
            usados.add(nuevo_valor)
            atraso = numero_a_atraso.get(nuevo_valor)
            if atraso is not None: atrasos_seleccionados[str(atraso)] += 1

        if len(seleccionados) == n_selecciones:
            # --- NUEVA VALIDACIÓN: Aplicar Regla 2 ---
            es_valida = True
            if historical_combinations:
                seleccionados_set = set([int(n) for n in seleccionados]) # Convertir a set de ints para comparar
                for hist_comb_set in historical_combinations:
                    if len(seleccionados_set.intersection(hist_comb_set)) > 2:
                        es_valida = False
                        break # Invalida, no es necesario seguir comparando
            
            if es_valida:
                seleccionados.sort(key=int)
                combinaciones.append(tuple(seleccionados))
    # ... resto del código sin cambios ...
    conteo_combinaciones = Counter(combinaciones)
    probabilidad_combinaciones = {}
    for combinacion, frecuencia in conteo_combinaciones.items():
        prob_comb = np.prod([distribucion_probabilidad.get(val, 0) for val in combinacion])
        probabilidad_combinaciones[combinacion] = (frecuencia, prob_comb)

    combinaciones_ordenadas = sorted(probabilidad_combinaciones.items(), key=lambda x: (-x[1][1], -x[1][0]))
    return combinaciones_ordenadas


def procesar_combinaciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, n_ejecuciones, historical_combinations):
    resultados_por_ejecucion = []
    # --- Pasar el historial a la función de generación ---
    task_args = (distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *task_args) for _ in range(n_ejecuciones)]
        for future in as_completed(futures):
            resultados_por_ejecucion.append(future.result())
    return resultados_por_ejecucion


def encontrar_combinaciones_coincidentes(resultados_por_ejecucion):
    if not resultados_por_ejecucion: return {}
    combinaciones_encontradas = {}
    for i, resultado in enumerate(resultados_por_ejecucion):
         for combinacion_tuple, _ in resultado:
              combinaciones_encontradas.setdefault(combinacion_tuple, set()).add(i)
    num_total_ejecuciones = len(resultados_por_ejecucion)
    return {
        comb: sorted(list(ejecuciones_set))
        for comb, ejecuciones_set in combinaciones_encontradas.items() if len(ejecuciones_set) == num_total_ejecuciones
    }

# --- Funciones para el Algoritmo Genético (Adaptadas) ---

def generar_individuo_deap(distribucion_prob, num_atraso, restr_atraso, n_sel):
    valores = list(distribucion_prob.keys())
    combinacion = []
    atrasos_seleccionados = Counter()
    usados = set()
    while len(combinacion) < n_sel:
        valores_posibles = [
            v for v in valores 
            if v not in usados and atrasos_seleccionados.get(str(num_atraso.get(v)), 0) < restr_atraso.get(str(num_atraso.get(v)), n_sel)
        ]
        if not valores_posibles: break
        nuevo_valor = random.choice(valores_posibles)
        combinacion.append(nuevo_valor)
        usados.add(nuevo_valor)
        atraso = num_atraso.get(nuevo_valor)
        if atraso is not None: atrasos_seleccionados[str(atraso)] += 1
    return creator.Individual(sorted(combinacion, key=int))


def evaluar_individuo_deap(individuo, distribucion_prob, num_atraso, restr_atraso, n_sel, historical_combinations):
    """Función de evaluación que ahora incluye la validación contra el historial."""
    if len(individuo) != n_sel: return (0,)
    
    # Validar restricciones de atraso
    atrasos_seleccionados = Counter([num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None])
    for atraso, cantidad in atrasos_seleccionados.items():
        if cantidad > restr_atraso.get(str(atraso), n_sel):
            return (0,)

    # --- NUEVA VALIDACIÓN: Aplicar Regla 2 contra el historial ---
    if historical_combinations:
        individuo_set = set([int(n) for n in individuo])
        for hist_comb_set in historical_combinations:
            if len(individuo_set.intersection(hist_comb_set)) > 2:
                return (0,) # Fitness cero si viola la regla
    
    # Si pasa todas las validaciones, calcular probabilidad
    probabilidad = np.prod([distribucion_prob.get(val, 0) for val in individuo])
    return (probabilidad,)


def ejecutar_algoritmo_genetico(n_generaciones, n_poblacion, cxpb, mutpb, distribucion_prob, numero_a_atraso, restricciones_atraso, historical_combinations, n_selecciones=6):
    toolbox = base.Toolbox()
    toolbox.register("individual", generar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # --- Pasar el historial a la función de evaluación ---
    toolbox.register("evaluate", evaluar_individuo_deap, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones, historical_combinations=historical_combinations)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=mutpb)
    toolbox.register("select", tools.selTournament, tournsize=3)

    try:
        population = toolbox.population(n=n_poblacion)
        if not population: return None, 0.0, "No se pudo generar una población inicial válida."
    except Exception as e:
        return None, 0.0, f"Error al crear la población inicial: {e}"

    algorithms.eaSimple(population, toolbox, cxpb, mutpb, n_generaciones, verbose=False)

    if population:
        best_ind = tools.selBest(population, k=1)[0]
        best_fitness = evaluar_individuo_deap(best_ind, distribucion_prob, numero_a_atraso, restricciones_atraso, n_selecciones, historical_combinations)[0]
        return best_ind, best_fitness, None
    else:
        return None, 0.0, "La población se volvió vacía durante la ejecución."


# ----------------------- Interfaz de Streamlit -----------------------

st.set_page_config(layout="wide")
st.title("Generador Avanzado de Combinaciones de Números")

# --- Carga de Datos ---
st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Datos de Atraso")
    uploaded_file = st.file_uploader("Sube tu archivo CSV (columnas 'Numero' y 'Atraso')", type="csv", key="atraso_uploader")
    df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset = load_data_and_counts(uploaded_file)

with col2:
    st.subheader("Historial de Combinaciones (Opcional)")
    historical_file = st.file_uploader("Sube un CSV con combinaciones anteriores (una por fila)", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(historical_file)

n_selecciones = 6

if df is not None:
     st.info(f"**Suma total de todos los 'Atraso' en el dataset cargado:** {total_atraso_dataset}")

# --- Configuración de Parámetros y Restricciones ---
st.header("2. Configurar Parámetros y Restricciones")

restricciones_finales = {}
if atrasos_disponibles_int:
    with st.expander("Configurar Restricciones de Atraso", expanded=True):
        st.write(f"Define la cantidad máxima de números permitida para cada valor de 'Atraso' en una combinación de {n_selecciones} números.")
        selected_atrasos_to_restrict = st.multiselect(
            "Selecciona los 'Atraso' a restringir:",
            options=[str(a) for a in atrasos_disponibles_int],
            default=[str(a) for a in atrasos_disponibles_int]
        )
        if selected_atrasos_to_restrict:
             cols = st.columns(4)
             for i, atraso_str in enumerate(selected_atrasos_to_restrict):
                with cols[i % 4]:
                    default_limit = atraso_counts.get(atraso_str, 0)
                    limit = st.number_input(
                        f"Max Atraso '{atraso_str}' (def: {default_limit}):",
                        min_value=0, max_value=n_selecciones, value=default_limit, step=1,
                        key=f"restriction_{atraso_str}"
                    )
                    restricciones_finales[atraso_str] = limit
        st.write("Restricciones configuradas:", restricciones_finales if restricciones_finales else "Ninguna")
else:
    st.info("Carga un archivo de 'Atraso' para configurar las restricciones.")

with st.expander("Configurar Parámetros de los Algoritmos"):
    col_ga, col_sim = st.columns(2)
    with col_ga:
        st.subheader("Algoritmo Genético")
        ga_ngen = st.slider("Número de Generaciones", 10, 1000, 200)
        ga_npob = st.slider("Tamaño de la Población", 100, 5000, 1000)
        ga_cxpb = st.slider("Prob. Cruce (CXPB)", 0.0, 1.0, 0.7, 0.05)
        ga_mutpb = st.slider("Prob. Mutación (MUTPB)", 0.0, 1.0, 0.2, 0.01)
    with col_sim:
        st.subheader("Simulación Concurrente")
        sim_n_combinaciones = st.number_input("Combinaciones por Ejecución", min_value=1000, value=100000, step=10000)
        sim_n_ejecuciones = st.number_input("Ejecuciones Concurrentes", min_value=1, value=8, step=1)

# --- Ejecución de los Algoritmos ---
st.header("3. Ejecutar Algoritmos")
if not numero_a_atraso or not distribucion_probabilidad:
    st.warning("Carga un archivo de Atrasos válido para poder ejecutar los algoritmos.")
else:
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Ejecutar Algoritmo Genético"):
            with st.spinner(f"Buscando la mejor combinación..."):
                mejor_individuo, mejor_fitness, error_msg = ejecutar_algoritmo_genetico(
                    ga_ngen, ga_npob, ga_cxpb, ga_mutpb,
                    distribucion_probabilidad, numero_a_atraso,
                    restricciones_finales, historical_combinations_set, n_selecciones
                )
            if error_msg: st.error(error_msg)
            elif mejor_individuo:
                st.subheader("Mejor Combinación (GA)")
                st.success(f"**Combinación: {' - '.join(map(str, mejor_individuo))}**")
                st.write(f"Fitness (Probabilidad): {mejor_fitness:.12f}")
                suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in mejor_individuo)
                st.write(f"Suma de Atrasos: {suma_atrasos}")
                valor_especial = total_atraso_dataset + 40 - suma_atrasos
                st.write(f"**Cálculo Especial:** {valor_especial}")
            else: st.warning("El GA no encontró una combinación válida.")

    with run_col2:
        if st.button("Ejecutar Simulación Concurrente"):
            with st.spinner("Generando y procesando combinaciones en paralelo..."):
                resultados_por_ejecucion = procesar_combinaciones(
                    distribucion_probabilidad, numero_a_atraso, restricciones_finales,
                    n_selecciones, sim_n_combinaciones, sim_n_ejecuciones, historical_combinations_set
                )
                combinaciones_coincidentes = encontrar_combinaciones_coincidentes(resultados_por_ejecucion)

            st.subheader("Combinaciones Coincidentes (Simulación)")
            if combinaciones_coincidentes:
                coincident_list = []
                first_run_map = dict(resultados_por_ejecucion[0]) if resultados_por_ejecucion else {}
                for comb_tuple, ejec_list in combinaciones_coincidentes.items():
                    prob = first_run_map.get(comb_tuple, (0, 0.0))[1]
                    suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in comb_tuple)
                    valor_especial = total_atraso_dataset + 40 - suma_atrasos
                    coincident_list.append({
                        "Combinación": " - ".join(map(str, comb_tuple)),
                        "Probabilidad": prob,
                        "Suma Atrasos": suma_atrasos,
                        "Cálculo Especial": valor_especial
                    })
                coincident_df = pd.DataFrame(coincident_list).sort_values("Probabilidad", ascending=False)
                st.dataframe(coincident_df)
            else:
                st.info("No se encontraron combinaciones que aparecieran en *todas* las simulaciones.")

# --- Sidebar ---
st.sidebar.header("Información")
st.sidebar.markdown("""
**Propósito:**
Generar combinaciones de números basándose en datos históricos de 'atraso', restricciones personalizadas y un historial de combinaciones anteriores.

**Archivos Requeridos:**
1.  **Datos de Atraso:** CSV con columnas `Numero` y `Atraso`.
2.  **Historial de Combinaciones (Opcional):** CSV donde cada fila contiene una combinación de números (ej. `1,2,3,4,5,6`). Este archivo activa la **Regla 2**: *las nuevas combinaciones no pueden compartir más de 2 números con ninguna combinación de este historial*.

**Cálculo Especial:**
`(Suma Atrasos Dataset) + 40 - (Suma Atrasos Combinación)`
""")
