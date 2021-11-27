CREATE TABLE public.car_manufacturer (
    car_manufacturer_id INT GENERATED ALWAYS AS IDENTITY,
    name VARCHAR(50),
    UNIQUE(name),
    PRIMARY KEY (car_manufacturer_id)
);


CREATE TABLE public.car (
    car_id INT GENERATED ALWAYS AS IDENTITY,
    car_manufacturer_id INT,
    model VARCHAR(50), 
    serial_number VARCHAR(50),
    weight INT,
    listprice INT,
    PRIMARY KEY (car_id),
    CONSTRAINT fk_car_manufacturer FOREIGN KEY(car_manufacturer_id) REFERENCES car_manufacturer(car_manufacturer_id) ON DELETE SET NULL 
);


CREATE TABLE public.customer (
    customer_id INT GENERATED ALWAYS AS IDENTITY,
    name VARCHAR(50),
    phone VARCHAR(50),
    UNIQUE(phone),
    PRIMARY KEY (customer_id)
);


CREATE TABLE public.salesperson (
    salesperson_id  INT GENERATED ALWAYS AS IDENTITY,
    name VARCHAR(50),
    phone VARCHAR(50),
    UNIQUE(phone),
    PRIMARY KEY (salesperson_id)
);


CREATE TABLE public.transaction (
    transact_dt timestamp,
    car_id INT,
    customer_id INT,
    salesperson_id INT,
    order_no VARCHAR(50),
    characteristics VARCHAR(50),
    sellingprice INT,
    added_dt timestamp DEFAULT NOW(),
    voided VARCHAR(1) default 'N',
    UNIQUE(order_no),
    CONSTRAINT fk_car FOREIGN KEY(car_id) REFERENCES car(car_id) ON DELETE SET NULL,
    CONSTRAINT fk_customer FOREIGN KEY(customer_id) REFERENCES customer(customer_id) ON DELETE SET NULL,
    CONSTRAINT fk_salesperson FOREIGN KEY(salesperson_id) REFERENCES salesperson(salesperson_id) ON DELETE SET NULL
);


insert into public.car_manufacturer(name) values ('Nissan');
insert into public.car_manufacturer(name) values ('Proton');
insert into public.car_manufacturer(name) values ('Toyota');
insert into public.car_manufacturer(name) values ('Honda');

insert into public.car (model,serial_number,weight,listprice) values ('Sunny','1231',1000, 10000);
insert into public.car (model,serial_number,weight,listprice) values ('Saga','1232',2000, 20000);
insert into public.car (model,serial_number,weight,listprice) values ('Moon','1233',3000, 30000);
insert into public.car (model,serial_number,weight,listprice) values ('Accord','1234',4000, 40000);

insert into public.customer(name,phone) values ('Lim','12345671');
insert into public.customer(name,phone) values ('Lee','12345672');
insert into public.customer(name,phone) values ('Wong','12345673');
insert into public.customer(name,phone) values ('Yap','12345674');

insert into public.salesperson(name,phone) values ('Andy','12345675');
insert into public.salesperson(name,phone) values ('John','12345676');
insert into public.salesperson(name,phone) values ('Peter','12345677');
insert into public.salesperson(name,phone) values ('Esther','12345678');

insert into public.transaction(transact_dt,car_id,customer_id,salesperson_id,order_no,sellingprice) values ('2021-10-01 14:01:10-08',1,1,1,'Order1',10000);
insert into public.transaction(transact_dt,car_id,customer_id,salesperson_id,order_no,sellingprice) values ('2021-11-01 15:01:10-08',2,2,2,'Order2',20000);
insert into public.transaction(transact_dt,car_id,customer_id,salesperson_id,order_no,sellingprice) values ('2021-11-15 16:01:10-08',3,3,3,'Order3',30000);
insert into public.transaction(transact_dt,car_id,customer_id,salesperson_id,order_no,sellingprice) values ('2021-10-27 17:01:10-08',4,4,4,'Order4',40000);




