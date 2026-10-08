"""Interval union and gaps, deliberately not GPU utilization or kernel launch counts."""
import statistics


def coverage(intervals):
    values=list(intervals)
    if not values:
        raise ValueError('Require attributed intervals')
    if any(type(start) is not int or type(duration) is not int or start < 0 or duration < 0
           for start,duration in values):
        raise ValueError('Require nonnegative integer timestamps and durations')
    values=sorted((s,s+d) for s,d in values)
    merged=[]
    for start,end in values:
        if merged and start <= merged[-1][1]:
            merged[-1][1]=max(end,merged[-1][1])
        else:
            merged.append([start,end])
    union=sum(end-start for start,end in merged)
    # The last interval by start need not have the greatest end.
    span=max(end for _,end in values)-values[0][0]
    gaps=[b[0]-a[1] for a,b in zip(merged,merged[1:])]
    return dict(interval_count=len(values),summed_duration_ns=sum(e-s for s,e in values),
        union_duration_ns=union,observed_span_ns=span,gap_count=len(gaps),
        total_gap_ns=sum(gaps),maximum_gap_ns=max(gaps,default=0),
        median_gap_ns=statistics.median(gaps) if gaps else None,
        interval_coverage_fraction=union/span if span else None,
        interpretation='Attributed interval coverage inside first-to-last span; not device utilization, launch count or isolated kernel time.')
