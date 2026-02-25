import requests
from pyproj import Transformer

# --- Konfigurasjon ---
NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
NVDB_POSISJON_URL = f"{NVDB_BASE_URL}/vegnett/api/v4/posisjon"
NVDB_OBJEKT_URL = f"{NVDB_BASE_URL}/vegobjekter/api/v4/vegobjekter/105"

NVDB_HEADERS = {
    "X-Client": "NTNU_EiT_StudentProject_Final_v4",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json"
}

# --- LOGIKK-VALG ---
# Sett denne til False for å bruke den gamle metoden (nærmeste vei uansett)
# Sett denne til True for å bruke "Smart" metode (holde seg på samme vei i kryss)
USE_SMART_LOGIC = False

# Global variabel for å huske hvilken vei vi sist var på (brukes av smart logikk)
LAST_VEGLENKE_ID = None

# Transformator fra GPS (WGS84) til NVDB (UTM33)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True)

def _fetch_fartsgrense_for_match(match):
    """Hjelpefunksjon for å hente fartsgrense-objektet fra NVDB for en spesifikk vei-match."""
    vls = match.get('veglenkesekvens', {})
    vls_id = vls.get('veglenkesekvensid')
    rel_pos = vls.get('relativPosisjon')
    vsys = match.get('vegsystemreferanse', {})
    veg_navn = vsys.get('kortform', 'Ukjent vei')
    distanse = match.get('avstand', 0)

    obj_params = {
        "veglenkesekvens": f"{rel_pos}@{vls_id}",
        "inkluder": "egenskaper",
        "srid": 5973
    }
    
    try:
        obj_resp = requests.get(NVDB_OBJEKT_URL, params=obj_params, headers=NVDB_HEADERS, timeout=5)
        if obj_resp.status_code == 200:
            obj_list = obj_resp.json().get("objekter", [])
            if obj_list:
                fart = None
                for e in obj_list[0].get("egenskaper", []):
                    if e["id"] == 2021:
                        fart = e["verdi"]
                        break
                if fart:
                    return {
                        "status": "ok",
                        "fartsgrense": int(fart),
                        "vei": veg_navn,
                        "veglenke_id": vls_id, # Lagrer denne for å huske veien
                        "avstand_meter": round(distanse, 1),
                        "full_info": f"{fart} km/t på {veg_navn}"
                    }
    except Exception:
        pass
    return None

def get_speed_limit_data(lat, lon):
    """
    Hovedfunksjon for å hente fartsgrense.
    Velger metode basert på konfigurasjon (Naive vs Smart).
    """
    global LAST_VEGLENKE_ID
    
    ost, nord = transformer.transform(lon, lat)
    pos_params = {
        "nord": nord, "ost": ost, "srid": 5973, 
        "maks_avstand": 40, "maks_antall": 5, "trafikantgruppe": "K"
    }
    
    try:
        pos_resp = requests.get(NVDB_POSISJON_URL, params=pos_params, headers=NVDB_HEADERS, timeout=10)
        pos_data = pos_resp.json()
        
        if not pos_data:
            return {"status":"error", "message": "Ingen vei funnet"}

        # --- METODE 1: SMART LOGIKK (Vei-lojalitet med forbedret sjekk) ---
        if USE_SMART_LOGIC and LAST_VEGLENKE_ID is not None:
            # Finn den forrige veien i de nye resultatene
            prioritert_match = next((m for m in pos_data if m.get('veglenkesekvens', {}).get('veglenkesekvensid') == LAST_VEGLENKE_ID), None)
            
            # Finn den absolutt nærmeste veien i de nye resultatene
            closest_match = pos_data[0]

            if prioritert_match:
                dist_gammel = prioritert_match.get('avstand', 100)
                dist_ny = closest_match.get('avstand', 0)
                
                # LOGIKK: 
                # Vi bytter vei hvis den nye veien er mer enn 10 meter nærmere enn den gamle.
                # Dette hindrer "hopping" ved støy, men tillater svinging i kryss.
                if dist_gammel < 30 and (dist_gammel - dist_ny) < 10:
                    resultat = _fetch_fartsgrense_for_match(prioritert_match)
                    if resultat:
                        return resultat
                    
        # --- METODE 2: NAIVE / STANDARD (Velg nærmeste som har fartsgrense) ---
        for match in pos_data:
            resultat = _fetch_fartsgrense_for_match(match)
            if resultat:
                # Oppdater hvilken vei vi er på nå slik at neste kall husker det
                LAST_VEGLENKE_ID = resultat.get("veglenke_id")
                return resultat

        return {"status":"error", "message": "Ingen fartsgrense funnet i nærheten"}
        
    except Exception as e:
        return {"status":"error", "message": str(e)}