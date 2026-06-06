import plotly.graph_objects as go

from plotly.subplots import make_subplots


class Plotter:
    def __init__(self):
        self.datas = {}

    def add(self, x, *y, name, mode="lines"):
        if isinstance(name, str):
            name = (name,)
        for y, name in zip(y, name):
            if name not in self.datas:
                self.datas[name] = ([], [], mode)
            self.datas[name][0].append(x)
            self.datas[name][1].append(y)

    def show(self, *groups, xtitle=None, ytitle=None, spacing=None):
        # https://plotly.com/python/subplots/
        # https://plotly.com/python/line-charts/#line-plot-with-goscatter

        if len(groups) == 0:
            groups = (tuple(self.datas.keys()),)

        if isinstance(ytitle, str):
            ytitle = (ytitle,)

        fig = make_subplots(
            rows=len(groups), cols=1, shared_xaxes=True, vertical_spacing=spacing
        )

        for i, names in enumerate(groups):
            if isinstance(names, str):
                names = (names,)
            for name in names:
                x, y, mode = self.datas[name]
                fig.add_trace(
                    go.Scatter(x=x, y=y, mode=mode, name=name), row=i + 1, col=1
                )
                fig.update_yaxes(
                    title_text=name if ytitle is None else ytitle[i], row=i + 1, col=1
                )

        if xtitle is not None:
            fig.update_xaxes(title_text=xtitle, row=len(groups), col=1)

        fig.show()
