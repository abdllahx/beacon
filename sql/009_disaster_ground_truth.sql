-- Per-disaster ground truth view: prefers GDIS (peer-reviewed, Rosvold & Buhaug 2021),
-- falls back to EM-DAT's own coordinates when GDIS doesn't cover the disaster.
-- gt_source flag tells us which we used so the eval report can stratify.

CREATE OR REPLACE VIEW disaster_ground_truth AS
SELECT
    e.id              AS emdat_event_id,
    e.dis_no,
    e.disaster_type,
    e.country,
    e.start_date,
    -- Centroid (point): GDIS-aggregated if any rows, else EM-DAT native
    COALESCE(
        (SELECT ST_Centroid(ST_Collect(g.point))
         FROM gdis_locations g WHERE g.emdat_dis_no = e.dis_no),
        e.point
    ) AS gt_centroid,
    -- Bbox: envelope of all GDIS points (multi-admin disasters), else EM-DAT bbox
    COALESCE(
        (SELECT ST_Envelope(ST_Collect(g.point))
         FROM gdis_locations g WHERE g.emdat_dis_no = e.dis_no),
        e.bbox
    ) AS gt_bbox,
    CASE
        WHEN EXISTS (SELECT 1 FROM gdis_locations g WHERE g.emdat_dis_no = e.dis_no)
        THEN 'gdis'
        ELSE 'emdat_self'
    END AS gt_source,
    (SELECT count(*) FROM gdis_locations g WHERE g.emdat_dis_no = e.dis_no) AS gdis_n_locations
FROM emdat_events e;
