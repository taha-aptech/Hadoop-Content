from mrjob.job import MRJob

class MRClimateAnomaly(MRJob):

    def mapper(self, _, line):
        parts = line.strip().split(",")  # assuming CSV: date,temp,humidity,label
        if len(parts) >= 2:
            date, temp = parts[0], parts[1]
            try:
                temp = float(temp)
                if temp > 40:   # threshold for anomaly
                    yield "Anomaly", 1
                else:
                    yield "Normal", 1
            except ValueError:
                pass   # skip bad rows

    def reducer(self, key, values):
        yield key, sum(values)

if __name__ == "__main__":
    MRClimateAnomaly.run()
