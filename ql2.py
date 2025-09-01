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

# ----------------------- Funciones de Carga y Procesamiento de Datos -----------------------

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

                atraso_stats = {
                    "min": df['Atraso'].min(), "max": df['Atraso'].max(), "mean": df['Atraso'].mean(),
                    "p25": df['Atraso'].quantile(0.25), "p75": df['Atraso'].quantile(0.75)
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
    if uploaded_file is not None:
        try:
            df_hist = pd.read_csv(uploaded_file, header=None)
            historical_sets = [set(pd.to_numeric(row, errors='coerce').dropna().astype(int)) for _, row in df_hist.iterrows()]
            historical_sets = [s for s in historical_sets if len(s) >= 6]
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
    if not historical_sets or not numero_a_atraso or total_atraso_dataset is None: return None
    special_calc_values = [total_atraso_dataset + 40 - sum(numero_a_atraso.get(str(num), 0) for num in comb_set if str(num) in numero_a_atraso) for comb_set in historical_sets]
    if not special_calc_values: return None
    return {"min": int(np.min(special_calc_values)), "max": int(np.max(special_calc_values)), "mean": int(np.mean(special_calc_values)), "std": int(np.std(special_calc_values))}

# --- NUEVO: Función de análisis de composición histórica ---
@st.cache_data
def analyze_historical_composition(historical_sets, numero_a_atraso, composicion_ranges):
    """Analiza la composición de atrasos de las combinaciones históricas."""
    if not historical_sets or not numero_a_atraso: return None

    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'

    composition_counts = Counter()
    for hist_comb in historical_sets:
        current_composition = Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_ranges) for num in hist_comb)
        fingerprint = (current_composition.get('caliente', 0), current_composition.get('tibio', 0),
                       current_composition.get('frio', 0), current_composition.get('congelado', 0))
        composition_counts[fingerprint] += 1
    
    return composition_counts if composition_counts else None

# (El resto de funciones de backend como `generar_combinaciones_con_restricciones`, `procesar_combinaciones`, etc. no necesitan cambios y se incluyen al final)
# ... [Código de backend completo al final] ...

# ----------------------- Interfaz Gráfica de Streamlit -----------------------

st.set_page_config(layout="wide", page_title="Generador de Combinaciones Avanzado")
st.title("Generador de Combinaciones y Sistema de Recomendación Estratégica")

# Inicializar session state para el botón de aplicar sugerencia
if 'suggested_composition' not in st.session_state:
    st.session_state.suggested_composition = None

# --- 1. Carga de Archivos ---
st.header("1. Cargar Archivos de Datos")
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Sube CSV de Atrasos ('Numero', 'Atraso')", type="csv", key="atraso_uploader")
    df, numero_a_atraso, dist_prob, atrasos_disp, atraso_counts, total_atraso, atraso_stats = load_data_and_counts(uploaded_file)
with col2:
    hist_file = st.file_uploader("Sube CSV con Historial de Combinaciones (Opcional)", type="csv", key="history_uploader")
    historical_combinations_set = load_historical_combinations(hist_file)

n_selecciones = 6
if df is not None:
     st.info(f"**Suma total de 'Atraso' en el dataset:** {total_atraso}")

# --- 2. Configuración de Parámetros ---
st.header("2. Configurar Parámetros y Restricciones")
restricciones_finales, composicion_rules = {}, {}
total_count_composition = 0

if df is not None:
    # (El expander de Restricciones de Atraso (Etapa 1) no cambia)
    with st.expander("Configurar Restricciones de Atraso (Etapa 1)"):
        selected_atrasos = st.multiselect("Selecciona 'Atraso' a restringir:", options=[str(a) for a in atrasos_disp], default=[str(a) for a in atrasos_disp])
        cols = st.columns(4)
        for i, atraso_str in enumerate(selected_atrasos):
            with cols[i % 4]:
                limit = st.number_input(f"Max Atraso '{atraso_str}':", 0, n_selecciones, atraso_counts.get(atraso_str, 0), key=f"res_{atraso_str}")
                restricciones_finales[atraso_str] = limit

    with st.expander("Filtro de Composición de Atrasos (Etapa 2)", expanded=True):
        st.info("Define rangos para categorizar los números, analiza el historial y especifica la composición deseada.")
        
        st.subheader("2.1. Definir Categorías de Atraso")
        max_atraso = atraso_stats.get("max", 100)
        c1, c2 = st.columns(2)
        with c1:
            range_caliente = st.slider("Rango 'Caliente'", 0, max_atraso, (0, int(atraso_stats.get("p25", 5))), key="r_hot")
            range_frio = st.slider("Rango 'Frío'", 0, max_atraso, (int(atraso_stats.get("p75", 15)), max_atraso - 1), key="r_cold")
        with c2:
            range_tibio = st.slider("Rango 'Tibio'", 0, max_atraso, (range_caliente[1] + 1, range_frio[0] -1), key="r_warm")
            min_congelado = st.number_input("Atraso mínimo 'Congelado'", value=max_atraso, key="r_icy")
        
        current_ranges = {'caliente': range_caliente, 'tibio': range_tibio, 'frio': range_frio, 'congelado': (min_congelado, 9999)}
        
        # --- NUEVO: Mostrar análisis de composición histórica ---
        if historical_combinations_set:
            st.subheader("2.2. Análisis de Composición Histórica")
            comp_analysis = analyze_historical_composition(historical_combinations_set, numero_a_atraso, current_ranges)
            if comp_analysis:
                most_common = comp_analysis.most_common(5)
                most_common_comp, freq = most_common[0]
                
                st.success(f"""**Composición más frecuente en el historial:**
                - **{most_common_comp[0]}** Calientes | **{most_common_comp[1]}** Tibios | **{most_common_comp[2]}** Fríos | **{most_common_comp[3]}** Congelados""")

                if st.button("Aplicar Composición Sugerida"):
                    st.session_state.suggested_composition = most_common_comp
                    st.rerun()
                
                with st.container():
                    df_comp = pd.DataFrame(most_common, columns=['Composición (C-T-F-C)', 'Frecuencia'])
                    df_comp['Probabilidad (%)'] = (df_comp['Frecuencia'] / sum(comp_analysis.values())) * 100
                    st.dataframe(df_comp)
            else:
                st.warning("No se pudo realizar el análisis de composición del historial.")

        st.subheader("2.3. Especificar Composición Deseada")
        
        # Usar valores de session_state si el botón fue presionado
        suggested = st.session_state.suggested_composition
        c3, c4, c5, c6 = st.columns(4)
        with c3: count_caliente = st.number_input("Nº Calientes", 0, n_selecciones, suggested[0] if suggested else 2, key="c_hot")
        with c4: count_tibio = st.number_input("Nº Tibios", 0, n_selecciones, suggested[1] if suggested else 2, key="c_warm")
        with c5: count_frio = st.number_input("Nº Fríos", 0, n_selecciones, suggested[2] if suggested else 2, key="c_cold")
        with c6: count_congelado = st.number_input("Nº Congelados", 0, n_selecciones, suggested[3] if suggested else 0, key="c_icy")

        total_count_composition = count_caliente + count_tibio + count_frio + count_congelado
        if total_count_composition != n_selecciones:
            st.warning(f"La suma de las cantidades ({total_count_composition}) debe ser igual a {n_selecciones}.")
        else:
            st.success("La composición es válida.")
            composicion_rules = {
                'ranges': current_ranges,
                'counts': {'caliente': count_caliente, 'tibio': count_tibio, 'frio': count_frio, 'congelado': count_congelado}
            }

    # (El resto de la UI, como 'Cálculo Especial' y 'Parámetros de Algoritmos', no cambia)
    # ... [Resto de la UI completo al final] ...

else:
    st.info("Carga un archivo de Atrasos para empezar a configurar los parámetros.")

# --- Código Backend Completo (para copiar y pegar) ---
# Todas las funciones de backend que no han sido mostradas arriba se incluyen aquí para que el script sea autocontenido.

def filtrar_por_composicion(combinaciones, numero_a_atraso, composicion_rules):
    def get_category(atraso, ranges):
        if ranges['caliente'][0] <= atraso <= ranges['caliente'][1]: return 'caliente'
        elif ranges['tibio'][0] <= atraso <= ranges['tibio'][1]: return 'tibio'
        elif ranges['frio'][0] <= atraso <= ranges['frio'][1]: return 'frio'
        elif atraso >= ranges['congelado'][0]: return 'congelado'
        return 'otro'
    combinaciones_filtradas = [comb for comb in combinaciones if all(Counter(get_category(numero_a_atraso.get(str(num), -1), composicion_rules['ranges']) for num in comb).get(cat, 0) == count for cat, count in composicion_rules['counts'].items())]
    return combinaciones_filtradas

def generar_combinaciones_con_restricciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations, total_atraso_dataset, special_calc_range):
    valores = list(distribucion_probabilidad.keys())
    combinaciones = []
    intentos_totales = 0
    max_intentos = n_combinaciones * 250
    while len(combinaciones) < n_combinaciones and intentos_totales < max_intentos:
        intentos_totales += 1
        seleccionados = random.sample(valores, n_selecciones)
        if any(Counter(numero_a_atraso.get(n, -1) for n in seleccionados)[int(atraso)] > limite for atraso, limite in restricciones_atraso.items()): continue
        suma_atrasos_comb = sum(numero_a_atraso.get(val, 0) for val in seleccionados)
        valor_especial = total_atraso_dataset + 40 - suma_atrasos_comb
        if not (special_calc_range[0] <= valor_especial <= special_calc_range[1]): continue
        if historical_combinations and any(len(set(int(n) for n in seleccionados).intersection(hist_set)) > 2 for hist_set in historical_combinations): continue
        seleccionados.sort(key=int)
        combinaciones.append(tuple(seleccionados))
    conteo = Counter(combinaciones)
    return sorted({c: (f, np.prod([distribucion_probabilidad.get(v, 0) for v in c])) for c, f in conteo.items()}.items(), key=lambda x: (-x[1][1], -x[1][0]))

def procesar_combinaciones(distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, n_ejecuciones, historical_combinations, total_atraso_dataset, special_calc_range):
    resultados = []
    args = (distribucion_probabilidad, numero_a_atraso, restricciones_atraso, n_selecciones, n_combinaciones, historical_combinations, total_atraso_dataset, special_calc_range)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generar_combinaciones_con_restricciones, *args) for _ in range(n_ejecuciones)]
        for future in as_completed(futures):
            try: resultados.append(future.result())
            except Exception as e: st.error(f"Un proceso falló: {e}")
    return resultados

def analizar_suma_especial_probabilidad(resultados, numero_a_atraso, total_atraso_dataset):
    if not resultados or not numero_a_atraso or total_atraso_dataset is None: return None
    counts, total_combs = Counter(), sum(f for res in resultados for _, (f, _) in res)
    if total_combs == 0: return None
    for res in resultados:
        for comb, (freq, _) in res:
            suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in comb)
            counts[total_atraso_dataset + 40 - suma_atrasos] += freq
    return counts, total_combs

def evaluar_individuo_deap(individuo, dist_prob, num_atraso, restr_atraso, n_sel, hist_combs, total_atraso, special_range):
    if len(individuo) != n_sel or len(set(individuo)) != n_sel: return (0,)
    atrasos = Counter(num_atraso.get(val) for val in individuo if num_atraso.get(val) is not None)
    if any(cantidad > restr_atraso.get(str(atraso), n_sel) for atraso, cantidad in atrasos.items()): return (0,)
    if hist_combs and any(len(set(int(n) for n in individuo).intersection(hist_set)) > 2 for hist_set in hist_combs): return (0,)
    suma_atrasos = sum(num_atraso.get(val, 0) for val in individuo)
    valor_especial = total_atraso + 40 - suma_atrasos
    if not (special_range[0] <= valor_especial <= special_range[1]): return (0,)
    return (np.prod([dist_prob.get(val, 0) for val in individuo]),)

def ejecutar_algoritmo_genetico(n_gen, n_pob, cxpb, mutpb, dist_prob, num_atraso, restr_atraso, hist_combs, total_atraso, special_range, n_sel=6):
    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, list(dist_prob.keys()), n_sel)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar_individuo_deap, dist_prob=dist_prob, num_atraso=num_atraso, restr_atraso=restr_atraso, n_sel=n_sel, hist_combs=hist_combs, total_atraso=total_atraso, special_range=special_range)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    pop = toolbox.population(n=n_pob)
    algorithms.eaSimple(pop, toolbox, cxpb, mutpb, n_gen, verbose=False)
    best = tools.selBest(pop, k=1)[0] if pop else None
    if best:
        best = sorted(list(set(best)), key=int)
        if len(best) != n_sel: return None, 0.0, "AG no mantuvo individuos válidos."
        return best, toolbox.evaluate(best)[0], None
    return None, 0.0, "Población final vacía."

# --- Resto de la UI ---
if df is not None:
    special_calc_range = (0, 99999)
    if historical_combinations_set and total_atraso is not None:
        with st.expander("Configurar Filtro de 'Cálculo Especial' (Etapa 1)"):
            stats = analyze_historical_special_calc(historical_combinations_set, total_atraso, numero_a_atraso)
            if stats:
                st.info(f"Historial: 'Cálculo Especial' varía de **{stats['min']}** a **{stats['max']}**, promedio **{stats['mean']}**.")
                default_range = (stats['mean'] - stats['std'], stats['mean'] + stats['std'])
                special_calc_range = st.slider("Rango deseado:", stats['min'] - 50, stats['max'] + 50, default_range)

    with st.expander("Configurar Parámetros de los Algoritmos"):
        col_ga, col_sim = st.columns(2)
        with col_ga: st.subheader("Algoritmo Genético"); ga_ngen=st.slider("Generaciones",10,1000,200); ga_npob=st.slider("Población",100,5000,1000); ga_cxpb=st.slider("Cruce",0.0,1.0,0.7); ga_mutpb=st.slider("Mutación",0.0,1.0,0.2)
        with col_sim: st.subheader("Simulación en Cascada"); sim_n_comb=st.number_input("Combinaciones/Ejec.",1000,value=50000); sim_n_ejec=st.number_input("Ejecuciones",1,value=8)

st.header("3. Ejecutar Algoritmos")
if df is None:
    st.warning("Carga un archivo de Atrasos para poder ejecutar los algoritmos.")
else:
    run_col1, run_col2 = st.columns(2)
    with run_col1:
        if st.button("Ejecutar Algoritmo Genético"):
            with st.spinner("Buscando la mejor combinación..."):
                mejor_ind, mejor_fit, err_msg = ejecutar_algoritmo_genetico(ga_ngen, ga_npob, ga_cxpb, ga_mutpb, dist_prob, numero_a_atraso, restricciones_finales, historical_combinations_set, total_atraso, special_calc_range, n_selecciones)
            if err_msg: st.error(err_msg)
            elif mejor_ind and mejor_fit > 0:
                st.subheader("Mejor Combinación (GA)"); st.success(f"**Combinación: {' - '.join(map(str, mejor_ind))}**")
                suma_atrasos = sum(numero_a_atraso.get(val, 0) for val in mejor_ind)
                st.write(f"**Cálculo Especial:** {total_atraso + 40 - suma_atrasos}")
            else: st.warning("El GA no encontró una combinación válida con las restricciones.")

    with run_col2:
        if st.button("Ejecutar Simulación en Cascada"):
            with st.spinner("Etapa 1: Generando combinaciones en paralelo..."):
                start_time = time.time()
                resultados = procesar_combinaciones(dist_prob, numero_a_atraso, restricciones_finales, n_selecciones, sim_n_comb, sim_n_ejec, historical_combinations_set, total_atraso, special_calc_range)
                st.info(f"Etapa 1 completada en {time.time() - start_time:.2f} segundos.")
            
            todas_unicas = list(set(tuple(int(n) for n in comb) for res in resultados for comb, _ in res))
            st.info(f"Se generaron **{len(todas_unicas)}** combinaciones únicas en la Etapa 1.")

            combinaciones_refinadas = []
            if total_count_composition == n_selecciones:
                with st.spinner("Etapa 2: Aplicando filtro de composición..."):
                    combinaciones_refinadas = filtrar_por_composicion(todas_unicas, numero_a_atraso, composicion_rules)
                st.success(f"Etapa 2 completada. Se encontraron **{len(combinaciones_refinadas)}** combinaciones que cumplen el perfil.")
            else:
                st.error("No se ejecutó Etapa 2: la suma de cantidades de composición no es correcta.")

            analisis_esp = analizar_suma_especial_probabilidad(resultados, numero_a_atraso, total_atraso)
            st.subheader("Análisis de 'Cálculo Especial' (Basado en Etapa 1)")
            if analisis_esp:
                counts, total_combs = analisis_esp
                if counts:
                    mas_probable, conteo = counts.most_common(1)[0]
                    st.metric("Valor más probable", int(mas_probable), f"{(conteo/total_combs)*100:.2f}% de probabilidad", delta_color="off")
                else: st.warning("No se generaron combinaciones para analizar.")
            else: st.warning("No se generaron combinaciones para analizar.")

            st.subheader(f"Resultados del Filtro de Composición (Etapa 2) - {len(combinaciones_refinadas)} combinaciones")
            if combinaciones_refinadas:
                data = [{"Combinación": " - ".join(map(str, sorted(c))), "Suma Atrasos": sum(numero_a_atraso.get(str(v),0) for v in c), "Cálculo Especial": total_atraso + 40 - sum(numero_a_atraso.get(str(v),0) for v in c)} for c in combinaciones_refinadas]
                st.dataframe(pd.DataFrame(data).reset_index(drop=True))
            else:
                st.info("Ninguna combinación superó el filtro de la Etapa 2.")

# --- Barra Lateral ---
st.sidebar.header("Guía del Modelo")
st.sidebar.markdown("""
Esta aplicación utiliza dos enfoques para generar combinaciones:

**1. Algoritmo Genético (AG):**
- Busca la **única mejor combinación** posible según un criterio de probabilidad teórica.
- Es un enfoque de **optimización** que converge a una solución.

**2. Simulación en Cascada:**
- **Etapa 1 (Generación):** Crea millones de combinaciones plausibles usando las reglas base.
- **Etapa 2 (Filtrado):** Selecciona de la Etapa 1 solo aquellas que coinciden con la **composición estratégica** (Caliente, Tibio, etc.) que has definido.
""")
st.sidebar.info("""
**NUEVO: Recomendación Estratégica**
- La aplicación ahora **analiza el historial** para decirte cuál ha sido la composición más frecuente en sorteos pasados.
- Usa el botón **"Aplicar Composición Sugerida"** para configurar tu filtro automáticamente con esta estrategia basada en datos.
""")
