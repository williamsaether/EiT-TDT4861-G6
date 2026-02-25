def calculate_weather_factor(temp, humidity, precipitation=0.0):
    """
    Beregner en vær-faktor (0.0 - 1.0) basert på atmosfæriske forhold.
    
    Antagelser:
    1. Is/Rim: Lav temp + høy fuktighet er farligere enn kun lav temp.
    2. Vannplaning: Mye regn (precipitation) krever lavere fart uansett sikt.
    3. Sikt: Ekstremt høy fuktighet (>95%) indikerer tåke/dis.
    """
    
    factor = 1.0
    risk_desc = "Optimale forhold"
    
    # --- 1. Is og Glatte veier (Vinterforhold) ---
    if temp <= 0:
        if humidity > 85:
            # Fare for underkjølt regn eller rimdannelse (Svart is)
            factor = 0.70
            risk_desc = "Høy risiko for is og rimdannelse"
        else:
            # Tørr kulde - mindre glatt, men fortsatt fare
            factor = 0.90
            risk_desc = "Lave temperaturer, fare for flekkvis glatte partier"
            
    elif 0 < temp <= 3:
        # Den "skumle" sonen hvor is smelter/fryser om hverandre
        if humidity > 80:
            factor = 0.85
            risk_desc = "Fare for slaps og snøsmelting"
            
    # --- 2. Regn og Vannplaning (Sommer/Høst) ---
    if precipitation > 0:
        if precipitation < 2.0:
            factor = min(factor, 0.90)
            risk_desc = "Lett regn, våt veibane"
        elif 2.0 <= precipitation < 5.0:
            factor = min(factor, 0.75)
            risk_desc = "Mye regn, fare for vannplaning"
        else:
            factor = min(factor, 0.60)
            risk_desc = "Kraftig regn, stor fare for vannplaning og redusert sikt"

    # --- 3. Tåke og Sikt (Luftfuktighet som proxy) ---
    if humidity > 96 and precipitation == 0:
        # Ved veldig høy fuktighet uten regn er det ofte tåke
        factor = min(factor, 0.80)
        risk_desc = "Høy luftfuktighet, fare for tåke og redusert sikt"

    # --- 4. Ekstrem kulde ---
    if temp < -15:
        # Biler/Dekk fungerer dårligere, fare for frosne bremser/vei
        factor = min(factor, 0.85)
        risk_desc = "Ekstrem kulde"

    return {
        "factor": round(factor, 2),
        "description": risk_desc,
        "is_hazardous": factor < 0.85
    }

# --- Test-eksempler ---
if __name__ == "__main__":
    print(f"Sol og varmt: {calculate_weather_factor(20, 40)}")
    print(f"Kaldt og rått: {calculate_weather_factor(-1, 92)}")
    print(f"Kraftig regn: {calculate_weather_factor(12, 90, 6.5)}")