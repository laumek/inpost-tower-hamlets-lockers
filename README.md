# InPost locker coverage for Tower Hamlets

I live in Limehouse, and my nearest InPost locker is about a 20-minute walk away. So I measured how many Tower Hamlets residents live within a 7-minute walk of a locker, and where five new lockers would help most.

Short version: about 41% of residents are within 7 minutes of a locker, or 57% if you count InPost's shop drop-off points too. Five well-placed lockers would reach around 52,000 more people, which would match today's lockers-plus-shops coverage, but available 24/7.

[Two-page write-up](report/InPost_Tower_Hamlets_Locker_Analysis.pdf)

![Map](outputs/coverage_map.png)

This is an independent analysis using public data. It isn't affiliated with InPost.

## How it works

Population comes from the 2021 Census for the borough's 913 Output Areas. Walking distances follow the actual street network rather than straight lines, which matters here: the canals, docks and the A12 turn a lot of short distances into long walks. New sites are picked one at a time from 359 candidates (supermarkets, convenience stores, fuel stations and rail stations), each time choosing the one that reaches the most people who aren't already covered.

The hardest part was the data. OpenStreetMap only had about a quarter of InPost's real lockers, so the results it gave were badly off. I rebuilt the locker list by hand from InPost's own locker finder, including lockers just over the borough boundary, since people near the edge use those too.

A few things to bear in mind. Lockers are placed at the centre of their postcode, so locations are approximate. Population stands in for parcel demand, which isn't the same thing. And the candidate sites come from map data, not a survey, so nobody has checked whether they have space for a locker.

## Running it

```bash
pip install -r requirements.txt
python tower_hamlets_lockers.py
```

You'll need two files in `data/`: the Census table TS001 for Tower Hamlets Output Areas (from Nomis, with area codes), and the 2021 Output Area boundaries from the ONS Open Geography Portal. The hand-collected locker list is already in `data/inpost_lockers_manual.csv`.

Data: ONS Census 2021 (OGL), OpenStreetMap contributors (ODbL), and InPost's public locker finder.
