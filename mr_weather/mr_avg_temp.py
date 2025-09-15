%%bash
cat > mr_avg_temp.py <<'PY'
from mrjob.job import MRJob
from mrjob.step import MRStep
import csv

class MRAvgTemp(MRJob):

    def steps(self):
        return [MRStep(mapper=self.mapper_get_temp,
                       combiner=self.combiner_sum,
                       reducer=self.reducer_avg)]

    def mapper_get_temp(self, _, line):
        # Use CSV reader
        row = next(csv.reader([line]))
        # Skip header
        if row[0] == "country":
            return
        country = row[0]
        try:
            temp = float(row[7])  # 8th column = temperature_celsius
        except:
            return
        yield country, (temp, 1)

    def combiner_sum(self, country, values):
        total, count = 0.0, 0
        for temp, c in values:
            total += temp
            count += c
        yield country, (total, count)

    def reducer_avg(self, country, values):
        total, count = 0.0, 0
        for temp, c in values:
            total += temp
            count += c
        if count > 0:
            yield country, total / count

if __name__ == "__main__":
    MRAvgTemp.run()
PY
