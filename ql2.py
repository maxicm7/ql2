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
# Se definen los objetos de DEAP una sola vez para evitar errores de recarga en Streamlit
if "FitnessMax" not in creator.__dict__:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMax)

# ----------------------- Funciones de Carga y Procesamiento de Datos -----------------------

@st.cache_data
def load_data_and_counts(uploaded_file):
    """Carga y procesa el archivo de atrasos, devolviendo estructuras de datos y estadísticas."""
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
                     st.warning("El archivo de atrasos no contiene filas válidas.")
                     return None, {}, {}, [], {}, 0, {}

                st.success("Archivo de atrasos cargado exitosamente.")
                numero_a_atraso = dict(zip(df['Numero'], df['Atraso']))
                atrasos_disponibles_int = sorted(list(set(df['Atraso'].tolist())))
                numeros_validos = list(numero_a_atraso.keys())
                prob_por_numero = 1.0 / len(numeros_validos) if numeros_validos else 0
                distribucion_probabilidad = {num: prob_por_numero for num in numeros_validos}
                atraso_counts = df['Atraso'].astype(str).value_counts().to_dict()
                total_atraso_dataset = df['Atraso'].sum()

                # Calcular estadísticas de atraso para los defaults de la UI
                atraso_stats = {
                    "min": df['Atraso'].min(),
                    "max": df['Atraso'].max(),
                    "mean": df['Atraso'].mean(),
                    "p25": df['Atraso'].quantile(0.25),
                    "p75": df['Atraso'].quantile(0.75)
                }

                return df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats
            else:
                 st.error("El archivo de atrasos debe contener las columnas 'Numero' y 'Atraso'.")
                 return None, {}, {}, [], {}, 0, {}
        except Exception as e:
            st.error(f"Error al procesar el archivo de atrasos: {e}")
            return None, {}, {}, [], {}, 0, {}
    return None, {}, {}, [], {}, 0, {}

@st.cache_data
def load_historical_combinations(uploaded_file):
    """Carga y procesa el archivo de historial de combinaciones."""
    if uploaded_file is not None:
        try:
            df_hist = pd.read_csv(uploaded_file, header=None)
            historical_sets = []
            for index, row in df_hist.iterrows():
                comb = pd.to_numeric(row, errors='coerce').dropna().astype(int).tolist()
                if len(comb) >= 6:
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

@st.cache_data
def analyze_historical_special_calc(historical_sets, total_atraso_dataset, numero_a_atraso):
    """Calcula la distribución del 'Cálculo Especial' para el set de datos histórico."""
    if not historical_sets or not numero_a_atraso or total_atraso_dataset is None: return None
    
    special_calc_values = []
    for comb_set in historical_sets:
        suma_atrasos_comb = sum(numero_a_atraso.get(str(num), 0) for num in comb_set if str(num) in numero_a_atraso)
        valor_especial = total_atraso_dataset + 40 - suma_atrasos_comb # El 40 es específico de la lotería 6/46
        special_calc_values.append(valor_especial)
        
    if not special_calc_values: return None
        
    return {
        "min": int(np.min(special_calc_values)), "max": int(np.max(special_calc_values)),
        "mean": int(np.mean(special_calc_values)), "std": int(np.std(special_calc_values))
    }

# ----------------------- Funciones del Modelo de Simulación (Etapa 1) -----------------------

def generar_combinaciones_con_restricciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations, total_atraso_dataset, special_calc_range):
    """Genera combinaciones mediante Monte Carlo con muestreo por rechazo."""
    valores = list(distribucion_probabilidad.keys())
    combinaciones = []
    intentos_totales = 0
    max_intentos = n_combinaciones * 250 # Aumentar el factor por si las restricciones son muy duras

    while len(combinaciones) < n_combinaciones and intentos_totales < max_intentos:
        intentos_totales += 1
        seleccionados = random.sample(valores, n_selecciones) # Muestreo más rápido
        
        # Validar restricciones
        atrasos_seleccionados = Counter(numero_a_atraso.get(n, -1) for n in seleccionados)
        if any(atrasos_seleccionados[int(atraso)] > limite for atraso, limite in restricciones_atraso.items()): continue

        suma_atrasos_comb = sum(numero_a_atraso.get(val, 0) for val in seleccionados)
        valor_especial = total_atraso_dataset + 40 - suma_atrasos_comb
        if not (special_calc_range[0] <= valor_especial <= special_calc_range[1]): continue
            
        if historical_combinations:
            seleccionados_set = set(int(n) for n in seleccionados)
            if any(len(seleccionados_set.intersection(hist_comb_set)) > 2 for hist_comb_set in historical_combinations): continue
        
        seleccionados.sort(key=int)
        combinaciones.append(tuple(seleccionados))

    conteo_combinaciones = Counter(combinaciones)
    probabilidad_combinaciones = {comb: (freq, np.prod([distribucion_probabilidad.get(val, 0) for val in comb])) for comb, freq in conteo_combinaciones.items()}
    return sorted(probabilidad_combinaciones.items(), key=lambda x: (-x[1][1], -x[1][0]))

def procesar_combinaciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, n_ejecuciones, historical_combinations, total_atraso_dataset, special_calc_range):
    """Ejecuta la generación de combinaciones en paralelo."""
    resultados = []
    task_args = (distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations, total_atraso_dataset, special_calc_range)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *task_args) for _ in range(n_ejecuciones)]
        for future in as_completed(futures):
            try: resultados.append(future.result())
            except Exception as e: st.error(f"Un proceso falló: {e}")
    return resultados

# ----------------------- Funciones del Modelo de Filtrado (Etapa 2) -----------------------

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    """Filtra una lista de combinaciones basada en la composición de sus atrasos."""
    
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'

    combinaciones_filtradas = []
    for comb in combinaciones:
        composition_counter = Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_rules['ranges']) for num in comb)
        
        # Comprobar si la composición coincide exactamente con la deseada
        if all(composition_counter.get(cat, 0) == count for cat, count in composicion_rules['counts'].items()):
            combinaciones_filtradas.append(comb)
            
    return combinaciones_filtradas

# ----------------------- Funciones del Algoritmo Genético y Análisis -----------------------

def analizar_suma_especial_probabilidad(resultados, numero_a_atraso, total_atraso_dataset):
    """Analiza los resultados de la Etapa 1 para encontrar el 'Cálculo Especial' más frecuente."""
    if not resultados or not numero_a_atraso or total_atraso_dataset is None: return None
    suma_especial_counts = Counter()
    total_combinaciones_generadas = sum(freq for res_proceso in resultados for _, (freq, _) in res_proceso)
    
    if total_combinaciones_generadas == 0: return None

    for res_proceso in resultados:
        for combinacion, (frecuencia, _) in res_proceso:
            suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in combinacion)
            valor_especial = total_atraso_dataset + 40 - suma_atrasos
            suma_especial_counts[valor_especial] += frecuencia
            
    return suma_especial_counts, total_combinaciones_generadas

def evaluar_individuo_deap(individuo, distribucion_prob, num_atraso, restr_atraso, n_sel, historical_combinations, total_atraso_dataset, special_calc_range):
    """Función de fitness para el Algoritmo Genético."""
    if len(individuo) != n_sel or len(set(individuo)) != n_sel: return (0,)
    
    atrasos = Counter(num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None)
    if any(cantidad > restr_atraso.get(str(atraso), n_sel) for atraso, cantidad in atrasos.items()): return (0,)

    if historical_combinations:
        individuo_set = set(int(n) for n in individuo)
        if any(len(individuo_set.intersection(hist_set)) > 2 for hist_set in historical_combinations): return (0,)
            
    suma_atrasos = sum(num_atraso.get(val, 0) for val in individuo)
    valor_especial = total_atraso_dataset + 40 - suma_atrasos
    if not (special_calc_range[0] <= valor_especial <= special_calc_range[1]): return (0,)

    probabilidad = np.prod([distribucion_prob.get(val, 0) for val in individuo])
    return (probabilidad,)

def ejecutar_algoritmo_genetico(n_generaciones, n_poblacion, cxpb, mutpb, distribucion_prob, numero_a_atraso, restricciones_atraso, historical_combinations, total_atraso_dataset, special_calc_range, n_selecciones=6):
    """Ejecuta el Algoritmo Genético completo."""
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, list(distribucion_prob.keys()), n_selecciones)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluar_individuo_deap, distribucion_prob=distribucion_prob, num_atraso=numero_a_atraso, restr_atraso=restricciones_atraso, n_sel=n_selecciones, historical_combinations=historical_combinations, total_atraso_dataset=total_atraso_dataset, special_calc_range=special_calc_range)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=n_poblacion)
    algorithms.eaSimple(population, toolbox, cxpb, mutpb, n_generaciones, verbose=False)
    
    best_ind = tools.selBest(population, k=1)[0] if population else None
    if best_ind:
        best_ind = sorted(list(set(best_ind)), key=int)
        if len(best_ind) != n_selecciones: return None, 0.0, "El AG no pudo mantener individuos válidos."
        best_fitness = toolbox.evaluate(best_ind)[0]
        return best_ind, best_fitness, None
    return None, 0.0, "La población final estaba vacía."

# ----------------------- Interfaz Gráfica de Streamlit -----------------------

st.set_page_config(layout="wide", page_title="Generador de Combinaciones Avanzado")
st.title("Generador Avanzado de Combinaciones")
st.markdown("Una herramienta para generar y refinar combinaciones plausibles utilizando simulación de Monte Carlo y Algoritmos Genéticos.")

# --- 1. Carga de Archivos ---
st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Sube CSV de Atrasos ('Numero', 'Atraso')", type="csv", key="atraso_uploader")
    df, numero_a_atraso, distribucion_probabilidad, atrasos_disponibles_int, atraso_counts, total_atraso_dataset, atraso_stats = load_data_and_counts(uploaded_file)
with col2:
    historical_file = st.file_uploader("Sube CSV con Historial de Combinaciones (Opcional)", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(historical_file)

n_selecciones = 6
if df is not None:
     st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso_dataset}")

# --- 2. Configuración de Parámetros ---
st.header("2. Configurar Parámetros y Restricciones")
restricciones_finales = {}
composicion_rules = {}
total_count_composition = 0

if df is not None:
    with st.expander("Configurar Restricciones de Atraso (Etapa 1)"):
        selected_atrasos = st.multiselect("Selecciona 'Atraso' a restringir:", options=[str(a) for a in atrasos_disponibles_int], default=[str(a) for a in atrasos_disponibles_int])
        cols = st.columns(4)
        for i, atraso_str in enumerate(selected_atrasos):
            with cols[i % 4]:
                limit = st.number_input(f"Max Atraso '{atraso_str}':", min_value=0, max_value=n_selecciones, value=atraso_counts.get(atraso_str, 0), key=f"res_{atraso_str}")
                restricciones_finales[atraso_str] = limit

    with st.expander("Filtro de Composición de Atrasos (Etapa 2)", expanded=True):
        st.info("Define rangos para categorizar los números y luego especifica cuántos números de cada categoría quieres en la combinación final.")
        
        st.subheader("2.1. Definir Categorías de Atraso")
        max_atraso = atraso_stats.get("max", 100)
        c1, c2 = st.columns(2)
        with c1:
            range_caliente = st.slider("Rango 'Caliente'", 0, max_atraso, (0, int(atraso_stats.get("p25", 5))), key="r_hot")
            range_frio = st.slider("Rango 'Frío'", 0, max_atraso, (int(atraso_stats.get("p75", 15)), max_atraso - 1), key="r_cold")
        with c2:
            range_tibio = st.slider("Rango 'Tibio'", 0, max_atraso, (range_caliente[1] + 1, int(atraso_stats.get("p75", 15)) -1), key="r_warm")
            min_congelado = st.number_input("Atraso mínimo 'Congelado'", value=max_atraso, key="r_icy")

        st.subheader("2.2. Especificar Composición Deseada")
        c3, c4, c5, c6 = st.columns(4)
        with c3: count_caliente = st.number_input("Nº Calientes", 0, n_selecciones, 2, key="c_hot")
        with c4: count_tibio = st.number_input("Nº Tibios", 0, n_selecciones, 2, key="c_warm")
        with c5: count_frio = st.number_input("Nº Fríos", 0, n_selecciones, 2, key="c_cold")
        with c6: count_congelado = st.number_input("Nº Congelados", 0, n_selecciones, 0, key="c_icy")

        total_count_composition = count_caliente + count_tibio + count_frio + count_congelado
        if total_count_composition != n_selecciones:
            st.warning(f"La suma de las cantidades ({total_count_composition}) debe ser igual a {n_selecciones}.")
        else:
            st.success("La composición es válida.")
            composicion_rules = {
                'ranges': {'caliente': range_caliente, 'tibio': range_tibio, 'frio': range_frio, 'congelado': (min_congelado, 9999)},
                'counts': {'caliente': count_caliente, 'tibio': count_tibio, 'frio': count_frio, 'congelado': count_congelado}
            }

    special_calc_range = (0, 99999) 
    if historical_combinations_set and total_atraso_dataset is not None:
        with st.expander("Configurar Filtro de 'Cálculo Especial' (Etapa 1)"):
            stats = analyze_historical_special_calc(historical_combinations_set, total_atraso_dataset, numero_a_atraso)
            if stats:
                st.info(f"Análisis del historial: 'Cálculo Especial' varía de **{stats['min']}** a **{stats['max']}**, con un promedio de **{stats['mean']}**.")
                default_range = (stats['mean'] - stats['std'], stats['mean'] + stats['std'])
                special_calc_range = st.slider("Rango deseado para 'Cálculo Especial':", min_value=stats['min'] - 50, max_value=stats['max'] + 50, value=default_range)

    with st.expander("Configurar Parámetros de los Algoritmos"):
        col_ga, col_sim = st.columns(2)
        with col_ga: st.subheader("Algoritmo Genético"); ga_ngen=st.slider("Generaciones",10,1000,200); ga_npob=st.slider("Población",100,5000,1000); ga_cxpb=st.slider("Cruce",0.0,1.0,0.7); ga_mutpb=st.slider("Mutación",0.0,1.0,0.2)
        with col_sim: st.subheader("Simulación Concurrente"); sim_n_comb=st.number_input("Combinaciones/Ejec.",1000,value=50000); sim_n_ejec=st.number_input("Ejecuciones",1,value=8)
else:
    st.info("Carga un archivo de Atrasos para empezar a configurar los parámetros.")

# --- 3. Ejecución y Resultados ---
st.header("3. Ejecutar Algoritmos")
if df is None:
    st.warning("Carga un archivo de Atrasos para poder ejecutar los algoritmos.")
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
        if st.button("Ejecutar Simulación en Cascada"):
            # Etapa 1: Generación masiva
            with st.spinner("Etapa 1: Generando y procesando combinaciones en paralelo..."):
                start_time = time.time()
                resultados = procesar_combinaciones(distribucion_probabilidad, numero_a_atraso, restricciones_finales, n_selecciones, sim_n_comb, sim_n_ejec, historical_combinations_set, total_atraso_dataset, special_calc_range)
                st.info(f"Etapa 1 completada en {time.time() - start_time:.2f} segundos.")
            
            todas_las_combinaciones_unicas = list(set(tuple(int(n) for n in comb) for res in resultados for comb, _ in res))
            st.info(f"Se generaron **{len(todas_las_combinaciones_unicas)}** combinaciones únicas en la Etapa 1.")

            # Etapa 2: Filtrado por composición
            combinaciones_refinadas = []
            if total_count_composition == n_selecciones:
                with st.spinner("Etapa 2: Aplicando filtro de composición..."):
                    combinaciones_refinadas = filtrar_por_composicion(todas_las_combinaciones_unicas, numero_a_atraso, composicion_rules)
                st.success(f"Etapa 2 completada. Se encontraron **{len(combinaciones_refinadas)}** combinaciones que cumplen con el perfil.")
            else:
                st.error("No se pudo ejecutar la Etapa 2: la suma de cantidades de la composición no es correcta.")

            # Visualización de resultados
            analisis_suma_esp = analizar_suma_especial_probabilidad(resultados, numero_a_atraso, total_atraso_dataset)
            st.subheader("Análisis de 'Cálculo Especial' (Basado en Etapa 1)")
            if analisis_suma_esp:
                suma_counts, total_combs = analisis_suma_esp
                if suma_counts:
                    suma_mas_probable, conteo = suma_counts.most_common(1)[0]
                    probabilidad = (conteo / total_combs) * 100
                    st.metric(label="Valor más probable del 'Cálculo Especial'", value=int(suma_mas_probable), delta=f"{probabilidad:.2f}% de las combinaciones válidas", delta_color="off")
                else: st.warning("No se generaron combinaciones válidas para analizar.")
            else: st.warning("No se generaron combinaciones válidas para analizar.")

            st.subheader(f"Resultados del Filtro de Composición (Etapa 2) - {len(combinaciones_refinadas)} combinaciones")
            if combinaciones_refinadas:
                data = []
                first_run_map = dict(resultados[0]) if resultados else {}
                for comb in combinaciones_refinadas:
                    suma_atrasos = sum(numero_a_atraso.get(str(val), 0) for val in comb)
                    valor_especial = total_atraso_dataset + 40 - suma_atrasos
                    data.append({
                        "Combinación": " - ".join(map(str, sorted(comb))),
                        "Probabilidad (aprox)": first_run_map.get(tuple(map(str, sorted(comb))), (0, 0.0))[1],
                        "Suma Atrasos": suma_atrasos,
                        "Cálculo Especial": valor_especial
                    })
                st.dataframe(pd.DataFrame(data).sort_values("Probabilidad (aprox)", ascending=False).reset_index(drop=True))
            else:
                st.info("Ninguna de las combinaciones generadas superó el filtro de composición de la Etapa 2.")

# --- Barra Lateral ---
st.sidebar.header("Guía del Modelo")
st.sidebar.markdown("""
Esta aplicación utiliza dos enfoques para generar combinaciones de números:

**1. Algoritmo Genético (AG):**
- Busca la **única mejor combinación** posible según un criterio de probabilidad teórica.
- Es un enfoque de **optimización** que converge a una solución.
- Útil para encontrar un candidato "campeón".

**2. Simulación en Cascada (Monte Carlo):**
- Es un proceso de dos etapas para **explorar y refinar**.
- **Etapa 1:** Genera millones de combinaciones en paralelo, descartando las que no cumplen las reglas básicas (atrasos, historial, "Cálculo Especial"). El objetivo es crear un gran universo de candidatos de alta calidad.
- **Etapa 2:** Aplica un **filtro estratégico** sobre los resultados de la Etapa 1. Puedes especificar la "personalidad" de la combinación que buscas (ej. 2 números "calientes", 3 "fríos", etc.) para reducir drásticamente el conjunto final a las combinaciones que se ajustan a tu estrategia.
""")
st.sidebar.markdown("---")
st.sidebar.info("El **'Cálculo Especial'** es una fórmula que predice el estado futuro del sistema (la suma total de atrasos) si una combinación dada resultara ganadora.")
