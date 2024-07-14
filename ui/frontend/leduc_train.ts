import { LitElement, PropertyValueMap, PropertyValues, css, html } from 'lit';
import { customElement, property } from 'lit/decorators.js';
import { map } from 'lit/directives/map.js';
import { createRef, Ref, ref } from 'lit/directives/ref.js';
import { io } from 'socket.io-client'
import Chart from 'chart.js/auto'

type UI_State = {

}

@customElement('leduc-train-ui')
class Leduc_Train_UI extends LitElement {
    @property({ type: Object }) state: UI_State = {
        // pl_type: players,
        // ui_game_state: ["GameNotStarted", []],
        // messages: []
    };

    // socket = io('/leduc_game')

    constructor() {
        super()
        // this.socket.on('update', (state : UI_State) => {
        //     this.state = state;
        // });
        // this.addEventListener('game', (ev) => {
        //     ev.stopPropagation();
        //     console.log(ev);
        //     this.socket.emit('update', (ev as CustomEvent<Game_Events>).detail);
        // })
    }

    static styles = css`
        :host {
            padding: 50px;
            display: flex;
            flex-direction: row;
            box-sizing: border-box;
            height: 100%;
            width: 100%;
            /* background-color: red; */
        }

        div {
            height: 60%;
            width: 60%;
            /* background-color: green; */
        }
    `

    inputRef: Ref<HTMLCanvasElement> = createRef();

    render() {
        return html`<div><canvas ${ref(this.inputRef)}></canvas></div>`
    }

    protected firstUpdated(_changedProperties: PropertyValues): void {
        const chart_container = this.inputRef.value;
        if (chart_container) {
            const labels = ["January", "February", "March", "April", "May", "June", "July"];
            const data = {
                labels: labels,
                datasets: [
                    {
                        label: '1st Dataset',
                        data: [65, 59, 80, 81, 56, 55, 40],
                        fill: false,
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: '2nd Dataset',
                        data: [65, 59, 80, 81, 56, 55, 40].map(x => x + 5),
                        fill: false,
                        borderColor: 'rgb(155, 92, 92)',
                        tension: 0.1
                    },
            ]
            };
            new Chart(
                chart_container, {
                    type: 'line',
                    data: data,
                }
            );
            // const data = [
            //     { year: 2010, count: 10 },
            //     { year: 2011, count: 20 },
            //     { year: 2012, count: 15 },
            //     { year: 2013, count: 25 },
            //     { year: 2014, count: 22 },
            //     { year: 2015, count: 30 },
            //     { year: 2016, count: 28 },
            // ];

            // new Chart(
            //     chart_container,
            //     {
            //         type: 'bar',
            //         data: {
            //             labels: data.map(row => row.year),
            //             datasets: [
            //                 {
            //                     label: 'Acquisitions by year',
            //                     data: data.map(row => row.count)
            //                 }
            //             ]
            //         }
            //     }
            // );
        }
    }
}
