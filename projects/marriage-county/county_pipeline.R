###############################################################################
#
#  MARRIAGE, RELIGION & POLITICS -- COUNTY-LEVEL PIPELINE (corrected)
#
#  Output:  county_dashboard_full.csv      (one row per county)
#           county_dashboard_full.geojson  (optional, for Tableau maps)
#
#  What changed from the previous version
#    1. Every ACS code is checked against the Census variable labels before
#       any data is pulled. A wrong code now stops the script with the real
#       table layout printed, instead of silently producing bad numbers.
#    2. Marital status (B12001) uses the correct codes, and "now married"
#       excludes separated, so the five categories sum to exactly 100%.
#    3. Cohabiting and single-parent households come from B11012
#       (Households by Type), which separates partnered from unpartnered
#       householders.
#    4. Added: median duration of current marriage (B12504), median age at
#       first marriage (B12007), median age (B01002), marriages and divorces
#       in the last year (B12501, B12503).
#    5. Politics is more than one number: lean in 2012/2016/2020/2024,
#       swing, third-party share, turnout, and volatility.
#    6. Connecticut: ACS 2022+ reports planning regions, which do not match
#       the election or religion files. CT rows are taken from ACS 2021
#       (2017-2021), the last vintage with the eight historical counties.
#    7. Dominant religious tradition considers all six traditions.
#
#  NOT RUN by the author of this file (no Census API access). The label
#  check in section 2 is the safety net: run it first and fix anything
#  it reports.
#
###############################################################################


# =============================================================================
# 0. SETUP
# =============================================================================

# install.packages(c("tidycensus", "tidyverse", "readxl", "janitor", "sf", "tigris"))

suppressPackageStartupMessages({
  library(tidycensus)
  library(tidyverse)
  library(readxl)
  library(janitor)
})

# One-time, in the console (not in this script):
#   census_api_key("YOUR_KEY", install = TRUE)
if (Sys.getenv("CENSUS_API_KEY") == "") {
  stop("No Census API key found. Run census_api_key('YOUR_KEY', install = TRUE) once, then restart R.")
}

acs_year        <- 2023    # ACS 5-year, 2019-2023
use_ct_legacy   <- TRUE    # use ACS 2021 for Connecticut (historical counties)
ct_legacy_year  <- 2021
write_geojson   <- TRUE

election_file   <- "countypres_2000-2024.csv"
religion_file   <- "U_S__Religion_Census_2020_County_File.csv"
rucc_file       <- "ruralurbancodes2023.xls"
nchs_file       <- "NCHSURCodes2013.csv"      # optional; skipped if missing

election_years  <- c(2012, 2016, 2020, 2024)

pct <- function(num, den) if_else(den > 0, num / den * 100, NA_real_)


# =============================================================================
# 1. ACS VARIABLE SPEC
#    name    = column name used in this script
#    code    = ACS variable
#    expect  = regex the Census LABEL must match   (NA = not checked)
#    concept = regex the Census TABLE TITLE must match (NA = not checked)
# =============================================================================

acs_spec <- tribble(
  ~name,                  ~code,         ~expect,                                         ~concept,

  # --- B12001 Sex by Marital Status (15+) ---
  "pop_15plus",           "B12001_001",  "^Estimate!!Total:?$",                           "Sex by Marital Status",
  "m_never",              "B12001_003",  "!!Male:?!!Never married$",                        NA,
  "m_now",                "B12001_004",  "!!Male:?!!Now married:?$",                        NA,   # includes separated
  "m_sep",                "B12001_007",  "!!Male:?!!Now married.*!!Separated$",             NA,
  "m_wid",                "B12001_009",  "!!Male:?!!Widowed$",                              NA,
  "m_div",                "B12001_010",  "!!Male:?!!Divorced$",                             NA,
  "f_never",              "B12001_012",  "Female:?!!Never married$",                      NA,
  "f_now",                "B12001_013",  "Female:?!!Now married:?$",                      NA,   # includes separated
  "f_sep",                "B12001_016",  "Female:?!!Now married.*!!Separated$",           NA,
  "f_wid",                "B12001_018",  "Female:?!!Widowed$",                            NA,
  "f_div",                "B12001_019",  "Female:?!!Divorced$",                           NA,

  # --- Marriage timing ---
  "marriage_duration",    "B12504_001",  "^Estimate!!Total:?$",                           "Median Duration of Current Marriage",
  "age_first_marr_m",     "B12007_001",  "first marriage.*!!Male$",                         "Median Age at First Marriage",
  "age_first_marr_f",     "B12007_002",  "first marriage.*!!Female$",                       NA,
  "median_age",           "B01002_001",  "Median age.*Total:?$",                          "Median Age by Sex",

  # --- Marital events in the last year (codes from memory: let the check confirm) ---
  "m_married_last_yr",    "B12501_005",  "!!Male:?!!Ever married:?!!Married last year$",    "Marriages in the Last Year",
  "f_married_last_yr",    "B12501_010",  "Female:?!!Ever married:?!!Married last year$",  NA,
  "m_divorced_last_yr",   "B12503_005",  "!!Male:?!!Ever married:?!!Divorced last year$",   "Divorces in the Last Year",
  "f_divorced_last_yr",   "B12503_010",  "Female:?!!Ever married:?!!Divorced last year$", NA,

  # --- B11012 Households by Type ---
  "hh_total",             "B11012_001",  "^Estimate!!Total:?$",                           "Households by Type",
  "hh_married",           "B11012_002",  "Married-couple household:?$",                   NA,
  "hh_cohab",             "B11012_005",  "Cohabiting couple household:?$",                NA,
  "hh_f_alone",           "B11012_009",  "Female householder.*!!Living alone$",           NA,
  "hh_f_kids",            "B11012_010",  "Female householder[^!]*!!With (own )?children.*under 18 years$", NA,
  "hh_m_alone",           "B11012_014",  "!!Male householder.*!!Living alone$",             NA,
  "hh_m_kids",            "B11012_015",  "!!Male householder[^!]*!!With (own )?children.*under 18 years$",   NA,

  # --- Fertility ---
  "women_15to50",         "B13016_001",  "^Estimate!!Total:?$",                           "Women 15 to 50",
  "women_birth",          "B13016_002",  "Women who had a birth in the past 12 months:?$", NA,

  # --- Economy ---
  "median_income",        "B19013_001",  "Median household income",                       NA,
  "pov_universe",         "B17001_001",  "^Estimate!!Total:?$",                           "Poverty Status",
  "pov_below",            "B17001_002",  "below poverty level:?$",                        NA,
  "gini",                 "B19083_001",  "Gini Index",                                    NA,
  "pop_16plus",           "B23025_001",  "^Estimate!!Total:?$",                           "Employment Status",
  "labor_force",          "B23025_002",  "!!In labor force:?$",                           NA,
  "civ_labor_force",      "B23025_003",  "!!Civilian labor force:?$",                     NA,
  "unemployed",           "B23025_005",  "!!Unemployed$",                                 NA,
  "home_value",           "B25077_001",  "Median value",                                  NA,
  "rent_burden_pct",      "B25071_001",  "Median gross rent as a percentage",             NA,

  # --- Citizen voting-age population (turnout denominator) ---
  "cvap",                 "B29001_001",  "^Estimate!!Total:?$",                           "Citizen, Voting-Age Population"
)

# --- B15003 Educational attainment (25+): codes _001 to _025 ---
edu_spec <- tibble(
  name    = sprintf("edu_%02d", 1:25),
  code    = sprintf("B15003_%03d", 1:25),
  expect  = NA_character_,
  concept = NA_character_
) %>%
  mutate(
    expect = case_when(
      name == "edu_01" ~ "^Estimate!!Total:?$",
      name == "edu_16" ~ "12th grade, no diploma$",
      name == "edu_17" ~ "Regular high school diploma$",
      name == "edu_21" ~ "Associate's degree$",
      name == "edu_22" ~ "Bachelor's degree$",
      name == "edu_25" ~ "Doctorate degree$",
      TRUE             ~ NA_character_
    ),
    concept = if_else(name == "edu_01", "Educational Attainment", NA_character_)
  )

acs_spec <- bind_rows(acs_spec, edu_spec)
stopifnot(!anyDuplicated(acs_spec$name), !anyDuplicated(acs_spec$code))


# =============================================================================
# 2. CHECK EVERY CODE AGAINST THE CENSUS LABELS  (run this before anything else)
# =============================================================================

validate_acs_spec <- function(spec, year) {
  lv <- load_variables(year, "acs5", cache = TRUE) %>%
    select(code = name, label, table_title = concept)

  chk <- spec %>%
    left_join(lv, by = "code") %>%
    mutate(
      found      = !is.na(label),
      ok_label   = is.na(expect)  | coalesce(str_detect(label, regex(expect, ignore_case = TRUE)), FALSE),
      ok_concept = is.na(concept) | coalesce(str_detect(table_title, regex(concept, ignore_case = TRUE)), FALSE),
      ok         = found & ok_label & ok_concept
    )

  bad <- chk %>% filter(!ok)

  if (nrow(bad) > 0) {
    cat(sprintf("\n*** ACS %d: %d variable(s) do not match their expected label ***\n", year, nrow(bad)))
    print(bad %>% select(name, code, expect, label, table_title), n = Inf, width = Inf)

    bad_tables <- unique(str_extract(bad$code, "^[^_]+"))
    cat("\nActual layout of the affected table(s):\n")
    lv %>%
      filter(str_extract(code, "^[^_]+") %in% bad_tables) %>%
      arrange(code) %>%
      print(n = Inf, width = Inf)

    stop("Fix the codes in acs_spec (section 1) and re-run.")
  }

  cat(sprintf("ACS %d: all %d variable codes match their expected labels.\n", year, nrow(spec)))
  invisible(chk)
}

validate_acs_spec(acs_spec, acs_year)
if (use_ct_legacy) validate_acs_spec(acs_spec, ct_legacy_year)


# =============================================================================
# 3. PULL ACS  (one request set for all tables)
# =============================================================================

pull_acs_wide <- function(year, state = NULL) {
  get_acs(
    geography = "county",
    variables = set_names(acs_spec$code, acs_spec$name),
    year      = year,
    survey    = "acs5",
    output    = "wide",
    state     = state
  )
}

acs_raw <- pull_acs_wide(acs_year) %>%
  filter(!str_starts(GEOID, "72")) %>%               # drop Puerto Rico (no election/religion match)
  mutate(acs_source_year = acs_year)

if (use_ct_legacy) {
  ct_raw <- pull_acs_wide(ct_legacy_year, state = "CT") %>%
    mutate(acs_source_year = ct_legacy_year)

  acs_raw <- acs_raw %>%
    filter(!str_starts(GEOID, "09")) %>%
    bind_rows(ct_raw)
}

# Estimates: strip the trailing "E" that tidycensus adds
acs_est <- acs_raw %>%
  select(GEOID, NAME, acs_source_year, all_of(paste0(acs_spec$name, "E"))) %>%
  rename_with(~ str_remove(.x, "E$"), all_of(paste0(acs_spec$name, "E"))) %>%
  rename(geoid = GEOID, county = NAME) %>%
  # Census "not available" sentinels (e.g. -666666666) -> NA
  mutate(across(where(is.numeric), ~ replace(.x, which(.x <= -222222222), NA)))

# Margins of error for the three medians (small counties are noisy)
acs_moe <- acs_raw %>%
  transmute(
    geoid                 = GEOID,
    marriage_duration_moe = marriage_durationM,
    age_first_marr_m_moe  = age_first_marr_mM,
    age_first_marr_f_moe  = age_first_marr_fM
  )


# =============================================================================
# 4. DERIVE ACS MEASURES
# =============================================================================

acs_data <- acs_est %>%
  left_join(acs_moe, by = "geoid") %>%
  mutate(
    # ---- Marital status: five categories that partition the 15+ population ----
    separated_n     = m_sep + f_sep,
    now_married_n   = (m_now - m_sep) + (f_now - f_sep),   # "now married, except separated"
    married_pct       = pct(now_married_n,       pop_15plus),
    never_married_pct = pct(m_never + f_never,   pop_15plus),
    separated_pct     = pct(separated_n,         pop_15plus),
    widowed_pct       = pct(m_wid + f_wid,       pop_15plus),
    divorced_pct      = pct(m_div + f_div,       pop_15plus),

    # ---- Marriage timing ----
    # marriage_duration: median years in CURRENT marriage, among people now
    #   married or separated. It is a snapshot, strongly tied to how old the
    #   county is -- always read it alongside median_age.
    age_first_marriage = (age_first_marr_m + age_first_marr_f) / 2,   # simple midpoint of the two medians
    age_first_marr_gap = age_first_marr_m - age_first_marr_f,

    # ---- Marital events in the last year, per 1,000 ----
    marriage_rate_per_1k = (m_married_last_yr + f_married_last_yr) / pop_15plus * 1000,
    divorce_rate_per_1k_married =
      if_else(m_now + f_now > 0,
              (m_divorced_last_yr + f_divorced_last_yr) /
                (m_now + f_now + m_divorced_last_yr + f_divorced_last_yr) * 1000,
              NA_real_),

    # ---- Households ----
    total_hh              = hh_total,
    married_couple_hh_pct = pct(hh_married,              hh_total),
    cohabiting_pct        = pct(hh_cohab,                hh_total),
    living_alone_pct      = pct(hh_f_alone + hh_m_alone, hh_total),
    single_parent_pct     = pct(hh_f_kids + hh_m_kids,   hh_total),   # no spouse/partner, own children < 18

    # ---- Fertility ----
    fertility_rate = if_else(women_15to50 > 0, women_birth / women_15to50 * 1000, NA_real_),

    # ---- Economy ----
    poverty_rate      = pct(pov_below,   pov_universe),
    unemployment_rate = pct(unemployed,  civ_labor_force),
    lfp_rate          = pct(labor_force, pop_16plus),

    # ---- Education (25+) ----
    edu_less_hs  = rowSums(across(num_range("edu_", 2:16,  width = 2))),
    edu_hs       = edu_17 + edu_18,
    edu_some     = edu_19 + edu_20 + edu_21,
    edu_ba       = edu_22,
    edu_grad     = edu_23 + edu_24 + edu_25,
    less_than_hs_pct   = pct(edu_less_hs, edu_01),
    hs_diploma_pct     = pct(edu_hs,      edu_01),
    some_college_pct   = pct(edu_some,    edu_01),
    bachelors_pct      = pct(edu_ba,      edu_01),
    grad_degree_pct    = pct(edu_grad,    edu_01),
    bachelors_plus_pct = pct(edu_ba + edu_grad, edu_01)
  )


# =============================================================================
# 5. ELECTIONS  (MIT Election Data + Science Lab, countypres_2000-2024.csv)
# =============================================================================

election_raw <- read_csv(election_file, show_col_types = FALSE) %>% clean_names()

stopifnot(all(c("year", "county_fips", "party", "candidatevotes") %in% names(election_raw)))
if (!"mode" %in% names(election_raw)) election_raw$mode <- "TOTAL"

missing_years <- setdiff(election_years, unique(election_raw$year))
if (length(missing_years) > 0) {
  stop("Election file is missing year(s): ", paste(missing_years, collapse = ", "))
}

election_long <- election_raw %>%
  filter(year %in% election_years, !is.na(county_fips), !is.na(candidatevotes)) %>%
  mutate(
    county_fips = str_pad(as.character(as.integer(county_fips)), 5, pad = "0"),
    party       = toupper(trimws(party)),
    mode        = toupper(coalesce(as.character(mode), "TOTAL"))
  ) %>%
  group_by(year, county_fips) %>%
  # Some states report by vote mode (early, absentee, ...). If a county has a
  # TOTAL row, use only that; otherwise add the modes up. Avoids double counting.
  filter(if (any(mode == "TOTAL")) mode == "TOTAL" else TRUE) %>%
  summarize(
    dem_votes   = sum(candidatevotes[party == "DEMOCRAT"]),
    rep_votes   = sum(candidatevotes[party == "REPUBLICAN"]),
    total_votes = sum(candidatevotes),
    .groups     = "drop"
  ) %>%
  mutate(
    dem_pct   = pct(dem_votes, total_votes),
    rep_pct   = pct(rep_votes, total_votes),
    other_pct = 100 - dem_pct - rep_pct,
    lean      = dem_pct - rep_pct              # positive = more Democratic
  )

election_data <- election_long %>%
  select(county_fips, year, dem_pct, rep_pct, other_pct, lean, total_votes) %>%
  pivot_wider(
    names_from  = year,
    values_from = c(dem_pct, rep_pct, other_pct, lean, total_votes),
    names_glue  = "{.value}_{year}"
  ) %>%
  rowwise() %>%
  mutate(lean_volatility = sd(c(lean_2012, lean_2016, lean_2020, lean_2024), na.rm = TRUE)) %>%
  ungroup() %>%
  mutate(
    # Headline partisan measure (kept under the old names for Tableau)
    dem_pct       = dem_pct_2020,
    rep_pct       = rep_pct_2020,
    partisan_lean = lean_2020,
    lean_category = case_when(
      is.na(partisan_lean) ~ NA_character_,
      partisan_lean >  20  ~ "Strong D",
      partisan_lean >   5  ~ "Lean D",
      partisan_lean >  -5  ~ "Swing",
      partisan_lean > -20  ~ "Lean R",
      TRUE                 ~ "Strong R"
    ),
    # Other political dimensions
    swing_2012_2016       = lean_2016 - lean_2012,     # the 2016 realignment
    swing_2016_2024       = lean_2024 - lean_2016,
    swing_2012_2024       = lean_2024 - lean_2012,
    third_party_pct_2016  = other_pct_2016
  )

# Note: Alaska reports by election district, not borough, so most Alaska
# rows will not match. That is expected.


# =============================================================================
# 6. RELIGION  (U.S. Religion Census 2020, county file from ARDA)
# =============================================================================

religion_raw <- read_csv(religion_file, show_col_types = FALSE) %>% clean_names()

# Column names as they appear in YOUR download (after clean_names()).
rel_cols <- c(
  totrate  = "totrate_2020",
  evanrate = "evanrate_2020",
  mainrate = "mprtrate_2020",
  bprtrate = "bprtrate_2020",
  cathrate = "cathrate_2020",
  orthrate = "orthrate_2020",
  othrrate = "othrate_2020"
)

rel_missing <- setdiff(c("fips", rel_cols), names(religion_raw))
if (length(rel_missing) > 0) {
  cat("Religion columns not found:", paste(rel_missing, collapse = ", "), "\n")
  cat("Columns containing 'rate' or 'fips' in the file:\n")
  print(names(religion_raw)[str_detect(names(religion_raw), "rate|fips")])
  stop("Edit rel_cols in section 6 to match the file.")
}

religion_data <- religion_raw %>%
  select(fips, all_of(rel_cols)) %>%
  filter(!is.na(fips)) %>%
  mutate(
    fips = str_pad(as.character(as.integer(fips)), 5, pad = "0"),
    # rates are adherents per 1,000 population
    total_adherent_pct      = totrate  / 10,    # can exceed 100 in a few counties
    evangelical_pct         = evanrate / 10,
    mainline_protestant_pct = mainrate / 10,
    black_protestant_pct    = bprtrate / 10,
    catholic_pct            = cathrate / 10,
    orthodox_pct            = orthrate / 10,
    other_religion_pct      = othrrate / 10,    # LDS, Jewish, Muslim, Hindu, etc. lumped together
    non_affiliated_pct      = pmax(100 - total_adherent_pct, 0)
  )

tradition_cols <- c(
  "Evangelical"         = "evangelical_pct",
  "Mainline Protestant" = "mainline_protestant_pct",
  "Black Protestant"    = "black_protestant_pct",
  "Catholic"            = "catholic_pct",
  "Orthodox"            = "orthodox_pct",
  "Other"               = "other_religion_pct"
)

trad_matrix <- as.matrix(religion_data[, tradition_cols])
trad_matrix[is.na(trad_matrix)] <- -Inf

religion_data <- religion_data %>%
  mutate(
    dominant_tradition = names(tradition_cols)[max.col(trad_matrix, ties.method = "first")],
    dominant_tradition = if_else(is.na(totrate), NA_character_, dominant_tradition),
    # share of ADHERENTS in the largest tradition: 100 = one tradition only
    dominant_share_of_adherents =
      if_else(total_adherent_pct > 0,
              apply(trad_matrix, 1, max) / total_adherent_pct * 100, NA_real_)
  ) %>%
  select(fips, total_adherent_pct, evangelical_pct, mainline_protestant_pct,
         black_protestant_pct, catholic_pct, orthodox_pct, other_religion_pct,
         non_affiliated_pct, dominant_tradition, dominant_share_of_adherents)


# =============================================================================
# 7. RURAL / URBAN
# =============================================================================

rucc_raw <- read_excel(rucc_file) %>% clean_names()

rucc_col <- names(rucc_raw)[str_detect(names(rucc_raw), "^rucc")][1]
fips_col <- names(rucc_raw)[str_detect(names(rucc_raw), "fips")][1]
if (is.na(rucc_col) || is.na(fips_col)) {
  stop("Could not find RUCC / FIPS columns. Columns are: ", paste(names(rucc_raw), collapse = ", "))
}

rucc_data <- rucc_raw %>%
  transmute(
    fips      = str_pad(as.character(as.integer(.data[[fips_col]])), 5, pad = "0"),
    rucc_code = as.integer(.data[[rucc_col]]),
    urban_rural_3tier = case_when(
      rucc_code <= 3 ~ "Metro",
      rucc_code <= 7 ~ "Nonmetro - Urban",
      rucc_code <= 9 ~ "Nonmetro - Rural"
    ),
    urban_rural_6tier = case_when(
      rucc_code == 1         ~ "Large Metro (1M+)",
      rucc_code == 2         ~ "Medium Metro (250K-1M)",
      rucc_code == 3         ~ "Small Metro (<250K)",
      rucc_code %in% c(4, 5) ~ "Nonmetro - Large Town",
      rucc_code %in% c(6, 7) ~ "Nonmetro - Small Town",
      rucc_code %in% c(8, 9) ~ "Nonmetro - Rural"
    ),
    is_metro = rucc_code <= 3
  ) %>%
  filter(!is.na(fips)) %>%
  distinct(fips, .keep_all = TRUE)

if (file.exists(nchs_file)) {
  nchs_data <- read_csv(nchs_file, show_col_types = FALSE) %>%
    clean_names() %>%
    transmute(
      fips       = str_pad(as.character(as.integer(fips_code)), 5, pad = "0"),
      nchs_code  = as.integer(x2013_code),
      nchs_label = case_when(
        nchs_code == 1 ~ "Large Central Metro",
        nchs_code == 2 ~ "Large Fringe Metro",
        nchs_code == 3 ~ "Medium Metro",
        nchs_code == 4 ~ "Small Metro",
        nchs_code == 5 ~ "Micropolitan",
        nchs_code == 6 ~ "Noncore (Rural)"
      )
    )
} else {
  nchs_data <- tibble(fips = character(), nchs_code = integer(), nchs_label = character())
}


# =============================================================================
# 8. MERGE
# =============================================================================

dashboard_data <- acs_data %>%
  left_join(election_data, by = c("geoid" = "county_fips")) %>%
  left_join(religion_data, by = c("geoid" = "fips")) %>%
  left_join(rucc_data,     by = c("geoid" = "fips")) %>%
  left_join(nchs_data,     by = c("geoid" = "fips")) %>%
  mutate(
    turnout_2020 = pct(total_votes_2020, cvap),
    turnout_2024 = pct(total_votes_2024, cvap)
  )

stopifnot(!anyDuplicated(dashboard_data$geoid))

dashboard_final <- dashboard_data %>%
  select(
    # Identifiers
    geoid, county, acs_source_year,

    # Marital status
    married_pct, never_married_pct, separated_pct, widowed_pct, divorced_pct,

    # Marriage timing and events
    marriage_duration, marriage_duration_moe,
    age_first_marriage, age_first_marr_m, age_first_marr_f, age_first_marr_gap,
    age_first_marr_m_moe, age_first_marr_f_moe,
    median_age,
    marriage_rate_per_1k, divorce_rate_per_1k_married,

    # Households
    total_hh, married_couple_hh_pct, single_parent_pct,
    living_alone_pct, cohabiting_pct,

    # Fertility
    fertility_rate,

    # Economy
    median_income, poverty_rate, gini,
    unemployment_rate, lfp_rate, home_value, rent_burden_pct,

    # Education
    less_than_hs_pct, hs_diploma_pct, some_college_pct,
    bachelors_pct, grad_degree_pct, bachelors_plus_pct,

    # Politics
    dem_pct, rep_pct, partisan_lean, lean_category,
    lean_2012, lean_2016, lean_2020, lean_2024,
    swing_2012_2016, swing_2016_2024, swing_2012_2024,
    third_party_pct_2016, lean_volatility,
    turnout_2020, turnout_2024,

    # Religion
    total_adherent_pct, evangelical_pct, mainline_protestant_pct,
    black_protestant_pct, catholic_pct, orthodox_pct,
    other_religion_pct, non_affiliated_pct,
    dominant_tradition, dominant_share_of_adherents,

    # Rural / urban
    rucc_code, urban_rural_3tier, urban_rural_6tier, is_metro,
    nchs_code, nchs_label
  )


# =============================================================================
# 9. CHECKS  (the first two stop the script if they fail)
# =============================================================================

cat("\n====== SANITY CHECKS ======\n")

marital_sum <- with(dashboard_final,
                    married_pct + never_married_pct + separated_pct + widowed_pct + divorced_pct)
cat(sprintf("Marital status sum: min %.3f, mean %.3f, max %.3f (should all be 100)\n",
            min(marital_sum, na.rm = TRUE), mean(marital_sum, na.rm = TRUE), max(marital_sum, na.rm = TRUE)))
stopifnot(all(abs(marital_sum - 100) < 0.01, na.rm = TRUE))

edu_sum <- with(dashboard_final,
                less_than_hs_pct + hs_diploma_pct + some_college_pct + bachelors_pct + grad_degree_pct)
cat(sprintf("Education sum:      min %.3f, mean %.3f, max %.3f (should all be 100)\n",
            min(edu_sum, na.rm = TRUE), mean(edu_sum, na.rm = TRUE), max(edu_sum, na.rm = TRUE)))
stopifnot(all(abs(edu_sum - 100) < 0.01, na.rm = TRUE))

cat("\nPlausibility (national figures are roughly: duration ~20 yrs, first marriage ~28-30, turnout ~55-70%):\n")
dashboard_final %>%
  summarize(across(
    c(marriage_duration, age_first_marr_m, age_first_marr_f, median_age,
      cohabiting_pct, single_parent_pct, divorce_rate_per_1k_married,
      partisan_lean, swing_2012_2024, turnout_2020, total_adherent_pct),
    list(min = ~ min(.x, na.rm = TRUE),
         med = ~ median(.x, na.rm = TRUE),
         max = ~ max(.x, na.rm = TRUE),
         n_missing = ~ sum(is.na(.x))),
    .names = "{.col}__{.fn}"
  )) %>%
  pivot_longer(everything(), names_to = c("variable", "stat"), names_sep = "__") %>%
  pivot_wider(names_from = stat, values_from = value) %>%
  print(n = Inf)

cat("\nJoin coverage:\n")
dashboard_final %>%
  summarize(
    counties     = n(),
    has_election = sum(!is.na(partisan_lean)),
    has_religion = sum(!is.na(total_adherent_pct)),
    has_rucc     = sum(!is.na(rucc_code)),
    has_duration = sum(!is.na(marriage_duration)),
    has_age_fm   = sum(!is.na(age_first_marriage))
  ) %>%
  pivot_longer(everything()) %>%
  print()

cat("\nCounties without election data, by state FIPS (02 = Alaska is expected):\n")
dashboard_final %>%
  filter(is.na(partisan_lean)) %>%
  count(state_fips = substr(geoid, 1, 2)) %>%
  print(n = Inf)

noisy <- dashboard_final %>% filter(marriage_duration_moe > 3)
cat(sprintf("\n%d counties have a marriage-duration margin of error above 3 years (small samples).\n",
            nrow(noisy)))


# =============================================================================
# 10. EXPORT
# =============================================================================

write_csv(dashboard_final, "county_dashboard_full.csv", na = "")
cat("\nCSV exported:", nrow(dashboard_final), "counties,", ncol(dashboard_final), "columns\n")

if (write_geojson) {
  library(sf)
  library(tigris)
  options(tigris_use_cache = TRUE)

  shapes <- counties(cb = TRUE, year = acs_year) %>%
    select(geoid = GEOID) %>%
    filter(!str_starts(geoid, "72"))

  if (use_ct_legacy) {
    ct_shapes <- counties(state = "CT", cb = TRUE, year = ct_legacy_year) %>%
      select(geoid = GEOID)
    shapes <- shapes %>%
      filter(!str_starts(geoid, "09")) %>%
      bind_rows(st_transform(ct_shapes, st_crs(shapes)))
  }

  dashboard_geo <- shapes %>% inner_join(dashboard_final, by = "geoid")
  # GeoJSON keeps full column names (shapefiles cut them to 10 characters)
  st_write(dashboard_geo, "county_dashboard_full.geojson", delete_dsn = TRUE, quiet = TRUE)
  cat("GeoJSON exported:", nrow(dashboard_geo), "counties\n")
}


# =============================================================================
# 11. FIRST LOOK: marriage duration and age at first marriage
#     vs. religion and politics
# =============================================================================
#
# County-level relationships describe places, not individuals: a link between
# evangelical share and early marriage across counties does not show that
# evangelicals marry earlier.

analysis <- dashboard_final %>%
  filter(!is.na(marriage_duration), !is.na(age_first_marriage),
         !is.na(partisan_lean), !is.na(total_adherent_pct))

outcomes   <- c("marriage_duration", "age_first_marriage")
predictors <- c("median_age",
                "evangelical_pct", "mainline_protestant_pct", "black_protestant_pct",
                "catholic_pct", "other_religion_pct", "non_affiliated_pct",
                "partisan_lean", "swing_2012_2016", "swing_2016_2024",
                "third_party_pct_2016", "turnout_2020",
                "bachelors_plus_pct", "median_income")

cat("\n====== CORRELATIONS (Pearson, unweighted) ======\n")
cor(analysis[, predictors], analysis[, outcomes], use = "pairwise.complete.obs") %>%
  round(2) %>%
  print()

# How many separate political dimensions are there in these counties?
pol_vars <- c("partisan_lean", "swing_2012_2016", "swing_2016_2024",
              "third_party_pct_2016", "turnout_2020", "lean_volatility")
pol_pca <- prcomp(na.omit(analysis[, pol_vars]), scale. = TRUE)
cat("\n====== POLITICS: principal components ======\n")
print(summary(pol_pca)$importance %>% round(2))
print(round(pol_pca$rotation[, 1:3], 2))

# Regressions, standardized so coefficients are comparable; weighted by
# households so tiny counties do not dominate.
std <- function(x) as.numeric(scale(x))

model_data <- analysis %>%
  mutate(across(all_of(c(outcomes, predictors)), std, .names = "z_{.col}"))

rhs <- paste(paste0("z_", predictors), collapse = " + ")

m_duration <- lm(as.formula(paste("z_marriage_duration ~", rhs)),
                 data = model_data, weights = total_hh)
m_age      <- lm(as.formula(paste("z_age_first_marriage ~", rhs)),
                 data = model_data, weights = total_hh)

cat("\n====== MARRIAGE DURATION (standardized, household-weighted) ======\n")
print(round(summary(m_duration)$coefficients, 3))
cat("\n====== AGE AT FIRST MARRIAGE (standardized, household-weighted) ======\n")
print(round(summary(m_age)$coefficients, 3))

cat("\n====== PIPELINE COMPLETE ======\n")
